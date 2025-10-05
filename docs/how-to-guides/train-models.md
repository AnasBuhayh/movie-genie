# How to Train Models

Quick reference for training models in Movie Genie.

## Quick Start

```bash
# Run complete pipeline (data + training + evaluation + web)
dvc repro

# Train specific model
dvc repro two_tower_training
dvc repro bert4rec_training

# Force retrain
dvc repro --force two_tower_training
```

## Training Individual Models

### Two-Tower Retrieval Model

```bash
# Via DVC (recommended)
dvc repro two_tower_training

# Direct training (for debugging)
.venv/bin/python scripts/train_two_tower.py \
  --config configs/two_tower.yaml \
  --mlflow-config configs/mlflow.yaml \
  --output-dir models/two_tower
```

**What gets trained:**
- User embeddings
- Item embeddings
- Two-tower architecture for retrieval

**Logged to MLflow:**
- Training/validation loss per epoch
- Recall@10, @50, @100
- Coverage@100
- Model parameters
- Training time

### BERT4Rec Ranking Model

```bash
# Via DVC
dvc repro bert4rec_training

# Direct training
.venv/bin/python scripts/train_bert4rec.py \
  --config configs/bert4rec.yaml \
  --mlflow-config configs/mlflow.yaml \
  --output-dir models/bert4rec
```

**What gets trained:**
- BERT-based sequence model
- Self-attention layers
- Sequential recommendation model

**Logged to MLflow:**
- Training/validation loss per epoch
- MRR, NDCG metrics
- Sequence length statistics
- Attention weights

## Training Options

### Force Retrain

Retrain even if nothing changed:

```bash
dvc repro --force two_tower_training
```

### Custom Config

Modify config before training:

```bash
# Edit config
vim configs/two_tower.yaml

# Train with new config
dvc repro two_tower_training
```

### Different Output Directory

```bash
.venv/bin/python scripts/train_two_tower.py \
  --config configs/two_tower.yaml \
  --mlflow-config configs/mlflow.yaml \
  --output-dir models/two_tower_v2
```

## Configuration

### Two-Tower Config

Edit `configs/two_tower.yaml`:

```yaml
model:
  embedding_dim: 128          # Change embedding size
  dropout: 0.1                # Regularization

training:
  batch_size: 1024           # Adjust for GPU memory
  learning_rate: 0.001       # Learning rate
  num_epochs: 20             # Training epochs
  early_stopping_patience: 3 # Early stopping

data:
  min_interactions: 5        # Filter users with < N interactions
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

### BERT4Rec Config

Edit `configs/bert4rec.yaml`:

```yaml
model:
  hidden_size: 128
  num_attention_heads: 4
  num_hidden_layers: 2
  max_sequence_length: 50

training:
  batch_size: 256
  learning_rate: 0.0001
  num_epochs: 50
```

## Monitoring Training

### View Progress

Training logs are output to console:

```bash
dvc repro two_tower_training

# Output:
# Epoch 1/20 - Train Loss: 0.234, Val Loss: 0.189
# Epoch 2/20 - Train Loss: 0.156, Val Loss: 0.145
# ...
```

### Check MLflow

During or after training:

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser
open http://localhost:5000
```

View:
- Loss curves
- Metric progression
- Parameter values
- Training time

### View Metrics

After training completes:

```bash
# JSON metrics
cat models/two_tower/metrics.json | jq

# DVC metrics
dvc metrics show
```

## Training Data

Models train on processed data:

- **User sequences**: `data/processed/sequences_with_metadata.parquet`
- **Content features**: `data/processed/content_features.parquet`
- **User interactions**: `data/processed/user_item_interactions.parquet`

If data is missing:

```bash
# Run data pipeline first
dvc repro sequential_processing content_features
```

## GPU Training

### Check GPU Availability

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Use GPU

GPU is used automatically if available. Training scripts detect and use CUDA.

For CPU-only:

```bash
export CUDA_VISIBLE_DEVICES=""
dvc repro two_tower_training
```

## Common Issues

### CUDA Out of Memory

Reduce batch size in config:

```yaml
training:
  batch_size: 512  # Reduce from 1024
```

### Training Takes Too Long

- Reduce `num_epochs`
- Use smaller `embedding_dim`
- Use smaller dataset (adjust `data/raw/` size)

### Model Not Loading in Backend

Check files exist:

```bash
ls -la models/two_tower/
# Should see:
# - pytorch_model.bin
# - config.json
# - metrics.json
```

### MLflow Not Logging

Ensure `--mlflow-config` is passed:

```yaml
# In dvc.yaml
cmd: >
  .venv/bin/python scripts/train_two_tower.py
  --mlflow-config configs/mlflow.yaml  # Must include this!
```

## After Training

### View Results

```bash
# Metrics
cat models/two_tower/metrics.json | jq

# MLflow UI
mlflow ui --port 5000

# Frontend dashboard
open http://localhost:8080/metrics
```

### Test Model

```bash
# Start backend (loads model)
FLASK_PORT=5001 python scripts/start_server.py

# Test API
curl -X POST http://localhost:5001/api/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 10}' | jq
```

### Evaluate Model

```bash
# Run evaluation stage
dvc repro integrated_evaluation

# View evaluation results
cat results/integrated_system_evaluation.json | jq
```

## Best Practices

### 1. Always Use DVC

```bash
# ✅ Recommended
dvc repro two_tower_training

# ❌ Avoid (unless debugging)
python scripts/train_two_tower.py ...
```

DVC ensures reproducibility and tracks dependencies.

### 2. Enable MLflow

Always pass `--mlflow-config` for experiment tracking.

### 3. Test Small First

Before full training:

```bash
# Edit config - reduce epochs
num_epochs: 2  # Instead of 20

# Quick test
dvc repro --force two_tower_training

# If works, reset epochs and train fully
```

### 4. Monitor Resources

```bash
# Watch GPU
watch -n 1 nvidia-smi

# Watch CPU/memory
top
```

## Next Steps

- [MLflow Integration](mlflow-integration.md) - Add MLflow to your model
- [Serving Recommendations](serve-recommendations.md) - Deploy model to API
- [DVC Pipeline](dvc-pipeline.md) - Understanding the pipeline
- [Model Evaluation](../machine-learning/evaluation.md) - Evaluate performance

## Additional Resources

- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
