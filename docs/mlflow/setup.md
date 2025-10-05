# MLflow Setup and Usage Guide

This guide explains how MLflow is integrated with Movie Genie for experiment tracking and model management.

## Overview

MLflow is integrated with the DVC pipeline to automatically track:
- Training hyperparameters
- Training metrics (loss curves, learning rates)
- Evaluation metrics (recall@k, coverage, etc.)
- Model artifacts
- Dataset statistics

## Quick Start

### 1. Train Models (MLflow Logging Automatic)

```bash
# Run DVC pipeline - MLflow logging happens automatically
dvc repro

# Force retrain specific model
dvc repro -f two_tower_training
dvc repro -f bert4rec_training
```

MLflow will automatically:
- Create experiment runs
- Log all parameters and metrics
- Save model artifacts
- Track training time and hardware

### 2. View Experiments in MLflow UI

```bash
# Start MLflow UI (in separate terminal)
mlflow ui --port 5000

# Open browser
open http://localhost:5000
```

The MLflow UI shows:
- All training runs with metrics
- Parameter comparison
- Loss curves and metric charts
- Model artifacts and versions

### 3. Access Metrics via API

```bash
# Get summary of all models
curl http://localhost:5001/api/models/summary

# List all runs for Two-Tower model
curl "http://localhost:5001/api/models/runs?model_name=two-tower&limit=10"

# Get specific run details
curl http://localhost:5001/api/models/runs/<RUN_ID>

# Compare multiple runs
curl -X POST http://localhost:5001/api/models/compare \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["run1", "run2", "run3"]}'
```

## Architecture

### MLflow + DVC Integration

```
┌─────────────────────────────────────┐
│     DVC Pipeline (dvc repro)        │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  train_two_tower.py          │  │
│  │  - Loads config              │  │
│  │  - Starts MLflow run         │──┼──┐
│  │  - Logs params/metrics       │  │  │
│  │  - Trains model              │  │  │
│  │  - Saves artifacts           │  │  │
│  └──────────────────────────────┘  │  │
│                                     │  │
│  ┌──────────────────────────────┐  │  │
│  │  train_bert4rec.py           │  │  │
│  │  - Loads config              │  │  │
│  │  - Starts MLflow run         │──┼──┤
│  │  - Logs params/metrics       │  │  │
│  │  - Trains model              │  │  │
│  │  - Saves artifacts           │  │  │
│  └──────────────────────────────┘  │  │
└─────────────────────────────────────┘  │
                                         │
                                         ▼
                              ┌────────────────────┐
                              │   MLflow Storage   │
                              │                    │
                              │  mlruns/           │
                              │  ├─ experiments    │
                              │  ├─ runs           │
                              │  └─ metrics        │
                              │                    │
                              │  mlartifacts/      │
                              │  └─ models         │
                              └────────────────────┘
                                         │
                     ┌───────────────────┴──────────────────┐
                     │                                      │
                     ▼                                      ▼
          ┌─────────────────────┐              ┌──────────────────────┐
          │   MLflow UI          │              │  Backend API         │
          │   (localhost:5000)   │              │  /api/models/*       │
          │                      │              │                      │
          │  - Browse runs       │              │  - List runs         │
          │  - Compare metrics   │              │  - Get metrics       │
          │  - View charts       │              │  - Compare models    │
          └─────────────────────┘              └──────────────────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────────┐
                                               │  React Frontend      │
                                               │  Model Dashboard     │
                                               └──────────────────────┘
```

## Configuration

### MLflow Config (`configs/mlflow.yaml`)

```yaml
mlflow:
  tracking_uri: "file:./mlruns"  # Local file storage
  artifact_location: "./mlartifacts"
  default_experiment_name: "movie-genie-models"

  # For production, use database backend:
  # tracking_uri: "postgresql://user:pass@host/mlflow_db"
```

### DVC Config (`dvc.yaml`)

Training stages automatically pass `--mlflow-config` flag:

```yaml
two_tower_training:
  cmd: python scripts/train_two_tower.py --mlflow-config configs/mlflow.yaml
  deps:
    - configs/mlflow.yaml  # MLflow config is a dependency
```

## What Gets Logged

### Two-Tower Model

**Parameters:**
- `embedding_dim`, `learning_rate`, `num_epochs`
- `batch_size`, `margin`, `validation_split`
- `negative_sampling_ratio`, `min_user_interactions`

**Metrics:**
- `train_loss`, `val_loss` (per epoch)
- `recall@10`, `recall@50`, `recall@100`
- `coverage@10`, `coverage@50`, `coverage@100`
- `num_users`, `num_movies`, `total_parameters`
- `training_time_seconds`

**Tags:**
- `model_type`: "retrieval"
- `model_name`: "two-tower"
- `framework`: "pytorch"
- `status`: "inactive" (changed to "active" when deployed)

**Artifacts:**
- PyTorch model weights
- Training history JSON
- Evaluation metrics JSON

### BERT4Rec Model

**Parameters:**
- `max_seq_len`, `hidden_dim`, `num_layers`, `num_heads`
- `learning_rate`, `weight_decay`, `batch_size`
- `mask_prob`, `validation_split`

**Metrics:**
- `train_loss`, `val_loss`, `learning_rate` (per epoch)
- `final_train_loss`, `final_val_loss`
- `num_epochs_trained`, `total_parameters`
- `training_time_seconds`

**Tags:**
- `model_type`: "ranking"
- `model_name`: "bert4rec"
- `framework`: "pytorch"

## Common Workflows

### 1. Train and Track New Model

```bash
# Train model - MLflow tracks automatically
dvc repro two_tower_training

# View results in MLflow UI
mlflow ui --port 5000

# Or via API
curl http://localhost:5001/api/models/runs?model_name=two-tower&limit=1 | jq
```

### 2. Compare Model Versions

```bash
# Train with different hyperparameters
# Edit configs/two_tower.yaml (change embedding_dim)
dvc repro -f two_tower_training

# Edit again (change learning_rate)
dvc repro -f two_tower_training

# Compare in MLflow UI
mlflow ui --port 5000
# → Go to "Compare" and select runs

# Or via API
curl -X POST http://localhost:5001/api/models/compare \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["<run_id_1>", "<run_id_2>"]}' | jq
```

### 3. Find Best Model

```bash
# Get all runs sorted by recall@10 (in MLflow UI)
# Or query via Python:

python3 << 'EOF'
from movie_genie.backend.app.services.mlflow_client import MLflowService

mlflow_service = MLflowService()
runs = mlflow_service.get_experiment_runs(max_results=100)

# Sort by recall@10
sorted_runs = sorted(runs,
                    key=lambda r: r['metrics'].get('recall@10', 0),
                    reverse=True)

print(f"Best model: {sorted_runs[0]['run_name']}")
print(f"Recall@10: {sorted_runs[0]['metrics']['recall@10']}")
print(f"Run ID: {sorted_runs[0]['run_id']}")
EOF
```

### 4. Promote Model to Production

```python
from movie_genie.backend.app.services.mlflow_client import MLflowService
import mlflow

mlflow_service = MLflowService()
run_id = "<best_model_run_id>"

# Update tag to mark as active
mlflow.set_tag("status", "active")

# Now API will return it in production models
# GET /api/models/production
```

## API Reference

### List All Runs
```bash
GET /api/models/runs?model_type=retrieval&limit=10
```

### Get Specific Run
```bash
GET /api/models/runs/{run_id}
```

### Get Metric History (for charts)
```bash
GET /api/models/runs/{run_id}/metrics/train_loss/history
```

### Compare Multiple Models
```bash
POST /api/models/compare
Body: {"run_ids": ["id1", "id2", "id3"]}
```

### Get Summary Statistics
```bash
GET /api/models/summary
```

Returns:
```json
{
  "total_runs": 25,
  "counts_by_type": {
    "retrieval": 12,
    "ranking": 13
  },
  "counts_by_status": {
    "inactive": 23,
    "active": 2
  },
  "latest_models": [...]
}
```

## Troubleshooting

### MLflow UI Not Starting

```bash
# Check if port 5000 is already in use
lsof -i :5000

# Use different port
mlflow ui --port 5050
```

### No Runs Showing Up

```bash
# Check MLflow tracking directory exists
ls -la mlruns/

# Check experiment was created
ls -la mlruns/0/  # 0 is default experiment ID

# Verify runs exist
find mlruns/ -name "meta.yaml"
```

### API Returns Empty Results

```bash
# Check MLflow service is initialized
curl http://localhost:5001/api/models/summary

# Check Flask logs for errors
tail -f logs/server.log

# Verify mlruns/ directory path in configs/mlflow.yaml
cat configs/mlflow.yaml | grep tracking_uri
```

### Training Not Logging to MLflow

```bash
# Check MLflow is imported in training script
grep "import mlflow" scripts/train_two_tower.py

# Check MLflow config exists
cat configs/mlflow.yaml

# Manually test MLflow
python3 << 'EOF'
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("test")
with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("test", 1)
    mlflow.log_metric("accuracy", 0.95)
print("✅ MLflow working!")
EOF

# Check mlruns/ for the test run
ls -la mlruns/
```

## Production Deployment

For production, use a database backend instead of file storage:

```yaml
# configs/mlflow.yaml (production)
mlflow:
  tracking_uri: "postgresql://mlflow_user:password@mlflow-db.example.com/mlflow"
  artifact_location: "s3://my-bucket/mlflow-artifacts"
```

Benefits:
- **Concurrent access**: Multiple users/processes can log simultaneously
- **Scalability**: Handle thousands of runs
- **Durability**: Database backups
- **Remote access**: Team members can view experiments

## Next Steps

- **Frontend Dashboard**: Build React UI to display metrics (Phase 3)
- **Automated Reports**: Generate weekly model performance reports
- **A/B Testing**: Compare models in production
- **Alerts**: Notify when model performance degrades
- **Cost Tracking**: Log training costs and compare ROI

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow with DVC](https://dvc.org/doc/user-guide/experiment-management/mlflow-integration)
