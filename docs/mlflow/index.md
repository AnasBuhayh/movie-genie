# MLflow Documentation

Complete guide to using MLflow for experiment tracking in Movie Genie.

## Overview

Movie Genie uses MLflow to track machine learning experiments, manage models, and monitor performance metrics. All training runs are automatically logged to MLflow when using the DVC pipeline.

## Quick Links

- [MLflow Setup Guide](setup.md) - Complete setup and configuration
- [Integration Summary](integration-summary.md) - Backend + Frontend integration
- [Frontend Dashboard](dashboard-implementation.md) - React metrics dashboard

## MLflow in Movie Genie

### Automatic Tracking

Every model training run automatically logs:

- **Parameters**: Hyperparameters, model config, data splits
- **Metrics**: Training/validation loss, evaluation metrics (recall, NDCG, coverage)
- **Artifacts**: Model weights, config files, training plots
- **Tags**: Model type, framework, deployment status

### Experiment Organization

```
mlruns/
├── 0/                          # Default experiment
├── two-tower-retrieval/        # Two-Tower experiments
├── bert4rec-ranking/           # BERT4Rec experiments
└── matrix-factorization/       # Matrix Factorization experiments
```

### Accessing MLflow

#### 1. MLflow UI (Detailed Exploration)

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

- Browse all experiments and runs
- Compare metrics across runs
- View training curves
- Download artifacts

#### 2. Backend API (Programmatic Access)

```bash
curl http://localhost:5001/api/models/summary | jq
```

Endpoints:
- `GET /api/models/runs` - List all runs
- `GET /api/models/runs/{id}` - Get run details
- `GET /api/models/summary` - Dashboard statistics
- `POST /api/models/compare` - Compare multiple runs

#### 3. Frontend Dashboard (Visual Interface)

Navigate to: `http://localhost:8080/metrics`

Features:
- Summary statistics
- Model comparison charts
- Filter by type/status
- Quick links to MLflow UI

## Getting Started

### 1. Train a Model

```bash
# All models log to MLflow automatically
dvc repro two_tower_training
```

### 2. View Results

```bash
# Start MLflow UI
mlflow ui --port 5000

# Or use backend API
curl http://localhost:5001/api/models/runs | jq

# Or visit frontend
open http://localhost:8080/metrics
```

### 3. Compare Models

```bash
# Train multiple configurations
dvc repro --force two_tower_training  # First run
# Edit config
dvc repro --force two_tower_training  # Second run

# Compare in UI or dashboard
```

## Documentation Pages

- **[MLflow Setup](setup.md)** - Installation, configuration, usage guide
- **[Integration Summary](integration-summary.md)** - Complete backend integration details
- **[Dashboard Implementation](dashboard-implementation.md)** - Frontend metrics dashboard

## Common Tasks

### View Latest Model Metrics

```bash
# Using MLflow CLI
mlflow runs list --experiment-name "two-tower-retrieval"

# Using API
curl "http://localhost:5001/api/models/runs?model_name=two-tower&limit=1" | jq
```

### Register a Model

```bash
# Automatic during training
# Or manual via UI/API
```

### Compare Two Runs

```bash
# Get run IDs
curl "http://localhost:5001/api/models/runs?limit=2" | jq '.data[].run_id'

# Compare
curl -X POST http://localhost:5001/api/models/compare \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["<id1>", "<id2>"]}' | jq
```

## Integration with DVC

MLflow works seamlessly with DVC:

```yaml
# dvc.yaml
two_tower_training:
  cmd: >
    .venv/bin/python scripts/train_two_tower.py
    --mlflow-config configs/mlflow.yaml  # MLflow enabled
  deps:
    - configs/mlflow.yaml  # Tracked dependency
```

When DVC runs training, MLflow automatically:
1. Creates experiment if needed
2. Starts new run
3. Logs all parameters and metrics
4. Saves model artifacts
5. Registers model in registry

## Next Steps

- Learn how to [add MLflow to your model](../how-to-guides/mlflow-integration.md)
- Explore the [API Reference](../backend-frontend/api-reference.md)
- Read about [Model Evaluation](../machine-learning/evaluation.md)

## Resources

- [MLflow Official Documentation](https://mlflow.org/docs/latest/)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
