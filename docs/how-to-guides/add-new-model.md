# How to Add a New Model

This guide walks you through the complete process of adding a new recommendation model to the Movie Genie system, from implementation to deployment.

## Prerequisites

- Python environment set up (see [Installation](../getting-started/installation.md))
- Understanding of PyTorch basics
- Familiarity with the project structure (see [Project Structure](../reference/project-structure.md))

## Overview

Adding a new model involves:

1. Creating the model class
2. Implementing the training script
3. Adding evaluation logic
4. Integrating with MLflow
5. Adding to DVC pipeline
6. Updating the backend API
7. Testing the model

## Step 1: Create the Model Class

Create a new file in `movie_genie/models/` for your model implementation.

### Example: Creating a Simple Matrix Factorization Model

```python
# movie_genie/models/matrix_factorization.py

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MatrixFactorizationModel(nn.Module):
    """
    Simple matrix factorization model for collaborative filtering.

    Args:
        num_users: Total number of users
        num_items: Total number of items
        embedding_dim: Dimension of user and item embeddings
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute predicted ratings.

        Args:
            user_ids: Tensor of user IDs [batch_size]
            item_ids: Tensor of item IDs [batch_size]

        Returns:
            predictions: Predicted ratings [batch_size]
        """
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embeddings(item_ids)  # [batch_size, embedding_dim]

        # Apply dropout
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)

        # Get biases
        user_b = self.user_bias(user_ids).squeeze(-1)  # [batch_size]
        item_b = self.item_bias(item_ids).squeeze(-1)  # [batch_size]

        # Compute dot product
        dot_product = (user_emb * item_emb).sum(dim=1)  # [batch_size]

        # Final prediction
        predictions = dot_product + user_b + item_b + self.global_bias

        return predictions

    def predict_batch(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Predict ratings for a batch of user-item pairs.

        Args:
            user_ids: User IDs
            item_ids: Item IDs
            device: Device to use

        Returns:
            predictions: Predicted ratings
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)

        with torch.no_grad():
            predictions = self.forward(user_ids, item_ids)

        return predictions

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """Get embedding for a specific user."""
        user_tensor = torch.tensor([user_id])
        return self.user_embeddings(user_tensor).squeeze(0)

    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """Get embedding for a specific item."""
        item_tensor = torch.tensor([item_id])
        return self.item_embeddings(item_tensor).squeeze(0)

    def save_pretrained(self, save_path: str):
        """Save model weights and config."""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

        # Save config
        config = {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "model_type": "matrix_factorization"
        }

        import json
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str) -> "MatrixFactorizationModel":
        """Load model from saved checkpoint."""
        import json
        import os

        # Load config
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)

        # Create model
        model = cls(
            num_users=config["num_users"],
            num_items=config["num_items"],
            embedding_dim=config["embedding_dim"]
        )

        # Load weights
        state_dict = torch.load(
            os.path.join(load_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)

        logger.info(f"Model loaded from {load_path}")
        return model
```

### Key Components to Include

1. **`__init__` method**: Initialize model architecture
2. **`forward` method**: Define forward pass
3. **`predict_batch` method**: Batch prediction for inference
4. **`save_pretrained` method**: Save model weights and config
5. **`from_pretrained` class method**: Load saved model
6. **Proper typing**: Use type hints for all methods
7. **Documentation**: Docstrings for all public methods

## Step 2: Create Configuration File

Create a YAML config file in `configs/` directory:

```yaml
# configs/matrix_factorization.yaml

model:
  type: "matrix_factorization"
  name: "mf-model"
  embedding_dim: 128
  dropout: 0.1

training:
  batch_size: 1024
  learning_rate: 0.001
  num_epochs: 20
  weight_decay: 0.0001
  early_stopping_patience: 3

optimizer:
  type: "adam"
  betas: [0.9, 0.999]

scheduler:
  type: "reduce_on_plateau"
  factor: 0.5
  patience: 2
  min_lr: 0.00001

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  min_sequence_length: 5

evaluation:
  metrics:
    - recall@10
    - recall@50
    - recall@100
    - ndcg@10
    - coverage@100
  batch_size: 512
```

## Step 3: Create Training Script

Create a training script in `scripts/`:

```python
# scripts/train_matrix_factorization.py

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from movie_genie.models.matrix_factorization import MatrixFactorizationModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """Load and prepare training data."""
    logger.info("Loading data...")

    # Load your data here
    # This is an example - adjust based on your data format
    data_path = Path("data/processed/user_item_interactions.parquet")
    df = pd.read_parquet(data_path)

    # Create user and item mappings
    unique_users = df['user_id'].unique()
    unique_items = df['movie_id'].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    # Map IDs to indices
    df['user_idx'] = df['user_id'].map(user_to_idx)
    df['item_idx'] = df['movie_id'].map(item_to_idx)

    # Split data
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'num_users': len(unique_users),
        'num_items': len(unique_items),
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx
    }


def create_dataloaders(data: dict, config: dict):
    """Create PyTorch DataLoaders."""
    batch_size = config['training']['batch_size']

    dataloaders = {}
    for split in ['train', 'val', 'test']:
        df = data[split]

        # Convert to tensors
        user_ids = torch.tensor(df['user_idx'].values, dtype=torch.long)
        item_ids = torch.tensor(df['item_idx'].values, dtype=torch.long)
        ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

        # Create dataset
        dataset = TensorDataset(user_ids, item_ids, ratings)

        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True
        )

    return dataloaders


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for user_ids, item_ids, ratings in dataloader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)

        # Forward pass
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    model,
    dataloaders,
    config: dict,
    device: torch.device,
    mlflow_run=None
):
    """Main training loop."""
    import mlflow

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr']
    )

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs = config['training']['num_epochs']

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device
        )

        # Validate
        val_loss = validate(model, dataloaders['val'], criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if mlflow_run:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, dataloader, data, config, device):
    """Evaluate the model on test set."""
    model.eval()

    # Implement your evaluation metrics here
    # This is a simple example
    criterion = nn.MSELoss()
    test_loss = validate(model, dataloader, criterion, device)

    metrics = {
        'test_loss': test_loss,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

    logger.info(f"Test Results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mlflow-config', type=str, help='Path to MLflow config')
    parser.add_argument('--output-dir', type=str, default='models/matrix_factorization')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup MLflow if configured
    mlflow_run = None
    if args.mlflow_config:
        import mlflow
        from movie_genie.utils.mlflow_helper import setup_mlflow

        mlflow_config = load_config(args.mlflow_config)
        setup_mlflow(mlflow_config)

        experiment_name = mlflow_config.get('experiment_name', 'matrix-factorization')
        mlflow.set_experiment(experiment_name)

        mlflow_run = mlflow.start_run(
            run_name=f"mf-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Log config
        mlflow.log_params({
            'embedding_dim': config['model']['embedding_dim'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'num_epochs': config['training']['num_epochs']
        })

        mlflow.set_tags({
            'model_type': 'ranking',
            'model_name': 'matrix-factorization',
            'framework': 'pytorch'
        })

    try:
        # Prepare data
        data = prepare_data(config)
        dataloaders = create_dataloaders(data, config)

        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        model = MatrixFactorizationModel(
            num_users=data['num_users'],
            num_items=data['num_items'],
            embedding_dim=config['model']['embedding_dim'],
            dropout=config['model']['dropout']
        ).to(device)

        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        model = train_model(model, dataloaders, config, device, mlflow_run)

        # Evaluate
        metrics = evaluate_model(
            model, dataloaders['test'], data, config, device
        )

        # Save model
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))

        # Save metrics
        metrics_path = output_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log to MLflow
        if mlflow_run:
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_artifact(str(metrics_path))

        logger.info(f"Training complete. Model saved to {output_path}")

    finally:
        if mlflow_run:
            mlflow.end_run()


if __name__ == '__main__':
    main()
```

## Step 4: Add to DVC Pipeline

Add a new stage to `dvc.yaml`:

```yaml
  matrix_factorization_training:
    cmd: >
      .venv/bin/python scripts/train_matrix_factorization.py
      --config configs/matrix_factorization.yaml
      --mlflow-config configs/mlflow.yaml
      --output-dir models/matrix_factorization
    deps:
      - scripts/train_matrix_factorization.py
      - movie_genie/models/matrix_factorization.py
      - configs/matrix_factorization.yaml
      - configs/mlflow.yaml
      - data/processed/user_item_interactions.parquet
    outs:
      - models/matrix_factorization:
          cache: false
    metrics:
      - models/matrix_factorization/metrics.json:
          cache: false
```

## Step 5: Integrate with Backend API

Update the backend to load and use your model:

```python
# movie_genie/backend/app/services/model_service.py

from movie_genie.models.matrix_factorization import MatrixFactorizationModel

class ModelService:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load all trained models."""
        # Load matrix factorization model
        mf_path = "models/matrix_factorization"
        if Path(mf_path).exists():
            self.models['mf'] = MatrixFactorizationModel.from_pretrained(mf_path)
            logger.info("Matrix Factorization model loaded")

    def get_mf_recommendations(
        self,
        user_id: int,
        k: int = 10
    ) -> List[int]:
        """Get recommendations using Matrix Factorization."""
        if 'mf' not in self.models:
            raise ValueError("MF model not loaded")

        model = self.models['mf']
        # Implement recommendation logic here
        # ...
        return recommendations
```

Add new API endpoint:

```python
# movie_genie/backend/app/api/recommendations_routes.py

@bp.route('/mf', methods=['POST'])
def get_mf_recommendations():
    """Get Matrix Factorization recommendations."""
    data = request.get_json()
    user_id = data.get('user_id')
    k = data.get('k', 10)

    try:
        recommendations = model_service.get_mf_recommendations(user_id, k)
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error getting MF recommendations: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
```

## Step 6: Test Your Model

Create tests in `tests/`:

```python
# tests/test_matrix_factorization.py

import pytest
import torch
from movie_genie.models.matrix_factorization import MatrixFactorizationModel


def test_model_initialization():
    """Test model can be initialized."""
    model = MatrixFactorizationModel(
        num_users=1000,
        num_items=500,
        embedding_dim=64
    )
    assert model.num_users == 1000
    assert model.num_items == 500


def test_forward_pass():
    """Test forward pass works."""
    model = MatrixFactorizationModel(num_users=100, num_items=50)

    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([10, 20, 30])

    predictions = model(user_ids, item_ids)

    assert predictions.shape == (3,)


def test_save_and_load(tmp_path):
    """Test saving and loading model."""
    model = MatrixFactorizationModel(num_users=100, num_items=50)

    save_path = tmp_path / "model"
    model.save_pretrained(str(save_path))

    loaded_model = MatrixFactorizationModel.from_pretrained(str(save_path))

    assert loaded_model.num_users == model.num_users
    assert loaded_model.num_items == model.num_items
```

Run tests:

```bash
pytest tests/test_matrix_factorization.py -v
```

## Step 7: Train and Deploy

1. **Train the model:**

```bash
# Train using DVC
dvc repro matrix_factorization_training

# Or train directly
python scripts/train_matrix_factorization.py \
  --config configs/matrix_factorization.yaml \
  --mlflow-config configs/mlflow.yaml
```

2. **Check MLflow:**

```bash
mlflow ui --port 5000
```

3. **Verify metrics:**

```bash
cat models/matrix_factorization/metrics.json
```

4. **Test API endpoint:**

```bash
# Start backend
python scripts/start_server.py

# Test endpoint
curl -X POST http://localhost:5001/api/recommendations/mf \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 10}'
```

## Checklist

Use this checklist to ensure you've completed all steps:

- [ ] Model class created in `movie_genie/models/`
- [ ] Config file created in `configs/`
- [ ] Training script created in `scripts/`
- [ ] Added to `dvc.yaml` pipeline
- [ ] Integrated with backend `ModelService`
- [ ] Added API endpoint
- [ ] Created unit tests
- [ ] Tested training locally
- [ ] Verified MLflow logging
- [ ] Tested API endpoint
- [ ] Updated documentation

## Common Issues

### CUDA Out of Memory

Reduce batch size in config:

```yaml
training:
  batch_size: 512  # Reduce from 1024
```

### Model Not Loading in Backend

Check file paths and permissions:

```bash
ls -la models/matrix_factorization/
# Should see: config.json, pytorch_model.bin
```

### MLflow Not Logging

Verify MLflow config is passed to training script:

```bash
python scripts/train_matrix_factorization.py \
  --config configs/matrix_factorization.yaml \
  --mlflow-config configs/mlflow.yaml  # Important!
```

## Next Steps

- [MLflow Integration Guide](mlflow-integration.md)
- [Serving Recommendations](serve-recommendations.md)
- [Model Evaluation](../machine-learning/evaluation.md)

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
