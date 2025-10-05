# How to Integrate Your Model with MLflow

This guide shows you how to add MLflow experiment tracking to your model training pipeline.

## Prerequisites

- Model training script ready (see [Adding a New Model](add-new-model.md))
- MLflow installed (`pip install mlflow`)
- Basic understanding of MLflow concepts (runs, experiments, metrics)

## Overview

MLflow integration involves:

1. Setting up MLflow configuration
2. Logging parameters and hyperparameters
3. Logging metrics during training
4. Logging artifacts (models, plots, data)
5. Registering models in MLflow Model Registry
6. Querying experiments via API

## Step 1: MLflow Configuration

MLflow is already configured in `configs/mlflow.yaml`. Review the settings:

```yaml
# configs/mlflow.yaml
tracking_uri: "file:./mlruns"  # Local file-based tracking

experiments:
  two_tower:
    name: "two-tower-retrieval"
    description: "Two-Tower model for retrieval"
    tags:
      model_type: "retrieval"
      framework: "pytorch"

  bert4rec:
    name: "bert4rec-ranking"
    description: "BERT4Rec for sequential ranking"
    tags:
      model_type: "ranking"
      framework: "pytorch"

# Add your model here
  matrix_factorization:
    name: "matrix-factorization"
    description: "Matrix Factorization for collaborative filtering"
    tags:
      model_type: "ranking"
      framework: "pytorch"

artifact_location: "./mlartifacts"

model_registry:
  enabled: true

default_tags:
  project: "movie-genie"
  environment: "development"

metric_definitions:
  training:
    - train_loss
    - val_loss
    - learning_rate
    - epoch_time

  evaluation:
    - recall_at_10
    - recall_at_50
    - recall_at_100
    - ndcg_at_10
    - coverage_at_100

  system:
    - num_users
    - num_items
    - total_parameters
    - training_time_seconds
```

## Step 2: Import MLflow in Your Training Script

Add MLflow imports to your training script:

```python
# scripts/train_your_model.py

import mlflow
import mlflow.pytorch  # For PyTorch models
# or mlflow.sklearn for scikit-learn
# or mlflow.tensorflow for TensorFlow

from pathlib import Path
import yaml
```

## Step 3: Setup MLflow Helper (Optional but Recommended)

Create a helper function for MLflow setup:

```python
# movie_genie/utils/mlflow_helper.py (if not exists)

import mlflow
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_mlflow(mlflow_config: dict):
    """
    Setup MLflow tracking with configuration.

    Args:
        mlflow_config: MLflow configuration dictionary
    """
    # Set tracking URI
    tracking_uri = mlflow_config.get('tracking_uri', 'file:./mlruns')
    mlflow.set_tracking_uri(tracking_uri)

    logger.info(f"MLflow tracking URI: {tracking_uri}")


def get_or_create_experiment(experiment_name: str, description: str = None, tags: dict = None):
    """
    Get existing experiment or create new one.

    Args:
        experiment_name: Name of the experiment
        description: Optional description
        tags: Optional tags

    Returns:
        experiment_id: ID of the experiment
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=f"mlartifacts/{experiment_name}",
            tags=tags or {}
        )
        logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    if description:
        mlflow.set_experiment_tag("mlflow.note.content", description)

    return experiment_id
```

## Step 4: Initialize MLflow in Training Script

Add MLflow initialization at the start of your training function:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mlflow-config', type=str, help='Path to MLflow config')
    parser.add_argument('--output-dir', type=str, default='models/your_model')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)

    # Setup MLflow if configured
    mlflow_enabled = args.mlflow_config is not None

    if mlflow_enabled:
        mlflow_config = load_config(args.mlflow_config)
        setup_mlflow(mlflow_config)

        # Get experiment config for your model
        exp_config = mlflow_config['experiments'].get('your_model', {})
        experiment_name = exp_config.get('name', 'your-model-default')

        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Start run
        from datetime import datetime
        run_name = f"your-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        logger.info(f"MLflow run started: {run_name}")

    try:
        # Your training code here
        train_model(config, mlflow_enabled)

    finally:
        if mlflow_enabled:
            mlflow.end_run()
            logger.info("MLflow run ended")
```

## Step 5: Log Parameters

Log all hyperparameters and configuration at the start:

```python
def train_model(config, mlflow_enabled):
    """Train model with MLflow logging."""

    # Log parameters
    if mlflow_enabled:
        # Model parameters
        mlflow.log_params({
            'model_type': config['model']['type'],
            'embedding_dim': config['model']['embedding_dim'],
            'dropout': config['model']['dropout'],
        })

        # Training parameters
        mlflow.log_params({
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'num_epochs': config['training']['num_epochs'],
            'weight_decay': config['training']['weight_decay'],
            'optimizer': config['optimizer']['type'],
        })

        # Data parameters
        mlflow.log_params({
            'train_split': config['data']['train_split'],
            'val_split': config['data']['val_split'],
            'test_split': config['data']['test_split'],
        })
```

## Step 6: Log Metrics During Training

Log metrics at each epoch:

```python
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, mlflow_enabled):
    """Train for one epoch with MLflow logging."""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        # Training code...
        loss = criterion(output, target)

        # Log batch-level metrics (optional)
        if mlflow_enabled and batch_idx % 100 == 0:
            mlflow.log_metric('batch_loss', loss.item(), step=epoch * len(dataloader) + batch_idx)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # Log epoch-level metrics
    if mlflow_enabled:
        mlflow.log_metrics({
            'train_loss': avg_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }, step=epoch)

    return avg_loss


def validate(model, dataloader, criterion, device, epoch, mlflow_enabled):
    """Validate with MLflow logging."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in dataloader:
            # Validation code...
            loss = criterion(output, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # Log validation metrics
    if mlflow_enabled:
        mlflow.log_metric('val_loss', avg_loss, step=epoch)

    return avg_loss
```

## Step 7: Log Evaluation Metrics

Log final evaluation metrics after training:

```python
def evaluate_model(model, test_dataloader, config, mlflow_enabled):
    """Evaluate model and log metrics to MLflow."""

    # Compute metrics (example)
    metrics = {
        'test_loss': compute_test_loss(model, test_dataloader),
        'recall_at_10': compute_recall_at_k(model, test_dataloader, k=10),
        'recall_at_50': compute_recall_at_k(model, test_dataloader, k=50),
        'recall_at_100': compute_recall_at_k(model, test_dataloader, k=100),
        'ndcg_at_10': compute_ndcg_at_k(model, test_dataloader, k=10),
        'coverage_at_100': compute_coverage(model, test_dataloader, k=100),
    }

    # Log all metrics
    if mlflow_enabled:
        mlflow.log_metrics(metrics)

        # Log system metrics
        mlflow.log_metrics({
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': get_model_size_mb(model),
        })

    return metrics
```

## Step 8: Log Model Artifacts

Save and log model files:

```python
def save_and_log_model(model, output_dir, config, metrics, mlflow_enabled):
    """Save model and log to MLflow."""

    # Save model locally
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))

    # Save metrics
    metrics_path = output_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    if mlflow_enabled:
        # Log model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=config['model']['name']  # Auto-register
        )

        # Log metrics file
        mlflow.log_artifact(str(metrics_path))

        # Log config
        config_path = output_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(str(config_path))

        logger.info(f"Model logged to MLflow")
```

## Step 9: Add Tags

Tag your runs for better organization:

```python
if mlflow_enabled:
    mlflow.set_tags({
        'model_type': 'ranking',  # or 'retrieval', 'embedding', 'hybrid'
        'model_name': 'your-model',
        'framework': 'pytorch',
        'dvc_pipeline': 'true',  # If using DVC
        'status': 'inactive',  # 'active' when deployed
    })
```

## Step 10: Important Metric Naming Rules

!!! warning "Metric Name Restrictions"
    MLflow has restrictions on metric names:

    - ❌ **NO** special characters: `@`, `#`, `$`, `%`, `&`
    - ✅ **USE** underscores instead: `recall_at_10` (not `recall@10`)
    - ✅ Letters, numbers, underscores, hyphens, dots, spaces are OK

**Correct metric names:**

```python
# ✅ Good
mlflow.log_metrics({
    'recall_at_10': 0.234,
    'recall_at_50': 0.456,
    'coverage_at_100': 0.678,
    'ndcg_at_10': 0.123,
})

# ❌ Bad - will cause errors!
mlflow.log_metrics({
    'recall@10': 0.234,  # @ not allowed
    'coverage@100': 0.678,  # @ not allowed
})
```

If your code generates metrics with `@`, convert them:

```python
# Convert @ to _at_
evaluation_results = {}
for k in [10, 50, 100]:
    recall_value = compute_recall(k)
    # Use _at_ instead of @
    evaluation_results[f'recall_at_{k}'] = recall_value  # ✅ Correct

# Or convert existing metrics
def sanitize_metric_name(name: str) -> str:
    """Convert metric names to MLflow-compatible format."""
    return name.replace('@', '_at_')

# Then use:
for metric_name, value in raw_metrics.items():
    safe_name = sanitize_metric_name(metric_name)
    mlflow.log_metric(safe_name, value)
```

## Step 11: Complete Example

Here's a complete training script with MLflow integration:

```python
# scripts/train_your_model.py

import argparse
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch

from movie_genie.models.your_model import YourModel
from movie_genie.utils.mlflow_helper import setup_mlflow

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mlflow-config', type=str, help='MLflow config')
    parser.add_argument('--output-dir', type=str, default='models/your_model')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup MLflow
    mlflow_enabled = args.mlflow_config is not None

    if mlflow_enabled:
        with open(args.mlflow_config) as f:
            mlflow_config = yaml.safe_load(f)

        setup_mlflow(mlflow_config)

        exp_config = mlflow_config['experiments']['your_model']
        mlflow.set_experiment(exp_config['name'])

        run_name = f"your-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        # Log all parameters
        mlflow.log_params({
            **config['model'],
            **config['training'],
            'optimizer': config['optimizer']['type'],
        })

        # Set tags
        mlflow.set_tags({
            **exp_config['tags'],
            **mlflow_config['default_tags'],
            'status': 'inactive',
        })

    try:
        # Prepare data
        train_loader, val_loader, test_loader = prepare_data(config)

        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YourModel(**config['model']).to(device)

        # Train
        for epoch in range(config['training']['num_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, epoch, mlflow_enabled)
            val_loss = validate(model, val_loader, epoch, mlflow_enabled)

            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Evaluate
        metrics = evaluate_model(model, test_loader, mlflow_enabled)

        # Save
        output_path = Path(args.output_dir)
        model.save_pretrained(str(output_path))

        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log to MLflow
        if mlflow_enabled:
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_artifact(str(output_path / 'metrics.json'))

            # Log run ID to metrics file for frontend
            metrics['mlflow_run_id'] = mlflow.active_run().info.run_id
            with open(output_path / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

        logger.info("Training complete!")

    finally:
        if mlflow_enabled:
            mlflow.end_run()


if __name__ == '__main__':
    main()
```

## Step 12: Add to DVC Pipeline

Update `dvc.yaml` to pass MLflow config:

```yaml
  your_model_training:
    cmd: >
      .venv/bin/python scripts/train_your_model.py
      --config configs/your_model.yaml
      --mlflow-config configs/mlflow.yaml
      --output-dir models/your_model
    deps:
      - scripts/train_your_model.py
      - movie_genie/models/your_model.py
      - configs/your_model.yaml
      - configs/mlflow.yaml  # Add this!
      - data/processed/
    outs:
      - models/your_model:
          cache: false
    metrics:
      - models/your_model/metrics.json:
          cache: false
```

## Step 13: View Results

### Option 1: MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### Option 2: Backend API

```bash
# Get summary
curl http://localhost:5001/api/models/summary | jq

# List runs
curl "http://localhost:5001/api/models/runs?model_name=your-model" | jq

# Get specific run
curl "http://localhost:5001/api/models/runs/<RUN_ID>" | jq
```

### Option 3: Frontend Dashboard

```bash
# Navigate to
http://localhost:8080/metrics
```

## Best Practices

### 1. Always Use Context Manager (Recommended)

```python
with mlflow.start_run(run_name=run_name):
    # All logging happens here
    mlflow.log_params(...)
    mlflow.log_metrics(...)
    # Run automatically ends when exiting context
```

### 2. Log in Batches for Performance

```python
# ❌ Slow - many API calls
for metric, value in metrics.items():
    mlflow.log_metric(metric, value)

# ✅ Fast - single API call
mlflow.log_metrics(metrics)
```

### 3. Use Step Parameter for Time Series

```python
# Log training progress over epochs
for epoch in range(num_epochs):
    mlflow.log_metric('train_loss', loss, step=epoch)
```

### 4. Add Descriptive Tags

```python
mlflow.set_tags({
    'model_type': 'retrieval',
    'dataset': 'movielens-25m',
    'experiment_purpose': 'hyperparameter_tuning',
    'notes': 'Testing new architecture',
})
```

### 5. Log System Information

```python
import platform
import torch

mlflow.log_params({
    'python_version': platform.python_version(),
    'pytorch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
})
```

## Troubleshooting

### MLflow UI Shows No Runs

```bash
# Check mlruns directory exists
ls -la mlruns/

# Check experiments
ls -la mlruns/0/  # Default experiment

# Verify tracking URI
echo $MLFLOW_TRACKING_URI
```

### Metrics Not Appearing

- Check metric names don't contain `@` or other special characters
- Verify `mlflow.log_metrics()` is called inside active run
- Check MLflow UI filters aren't hiding runs

### Model Not Registering

```python
# Explicitly register model
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="your-model-name"  # Add this!
)
```

### Run ID Not in metrics.json

Make sure to save run ID:

```python
if mlflow_enabled:
    run_id = mlflow.active_run().info.run_id
    metrics['mlflow_run_id'] = run_id

    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
```

## Next Steps

- [Adding a New Model](add-new-model.md)
- [Serving Recommendations](serve-recommendations.md)
- [MLflow Setup Guide](../mlflow/mlflow-setup.md)
- [Model Metrics Dashboard](../backend-frontend/ml-integration.md)

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
