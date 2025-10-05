# MLflow Integration Summary

Complete summary of the MLflow + DVC + React integration for experiment tracking and model metrics visualization.

## Implementation Status

### Backend ✅ Complete

- [x] MLflow configuration (`configs/mlflow.yaml`)
- [x] Training scripts with MLflow logging
- [x] DVC pipeline integration
- [x] MLflow client service
- [x] REST API routes for model metrics
- [x] Enhanced startup logging

### Frontend ✅ Complete

- [x] TypeScript types for model metrics
- [x] API client service
- [x] React hooks for data fetching
- [x] Model metrics card component
- [x] Model comparison chart component
- [x] Complete dashboard
- [x] Route integration (`/metrics`)
- [x] Navigation from main page

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              DVC Pipeline (dvc repro)                │
│                                                      │
│  ┌──────────────────┐      ┌──────────────────┐   │
│  │ train_two_tower  │      │ train_bert4rec   │   │
│  │   + MLflow       │      │   + MLflow       │   │
│  └────────┬─────────┘      └────────┬─────────┘   │
│           │                         │              │
└───────────┼─────────────────────────┼──────────────┘
            │                         │
            └──────────┬──────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   MLflow Storage     │
            │   - mlruns/          │
            │   - mlartifacts/     │
            │   - mlflow.db        │
            └──────────┬───────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐      ┌─────────────────────┐
│  MLflow UI      │      │  Flask Backend      │
│  :5000          │      │  :5001              │
│                 │      │                     │
│  - Browse runs  │      │  /api/models/       │
│  - Compare      │      │  - runs             │
│  - Charts       │      │  - compare          │
│  - Registry     │      │  - summary          │
└─────────────────┘      └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  React Frontend     │
                         │  :8080              │
                         │                     │
                         │  - Metrics Dashboard│
                         │  - Model Comparison │
                         │  - Loss Charts      │
                         └─────────────────────┘
```

## What Gets Logged

### Every Training Run Logs:

**Parameters:**
- All hyperparameters (learning_rate, batch_size, etc.)
- Model architecture config
- Data processing config
- Optimizer settings

**Metrics:**
- Training loss (per epoch)
- Validation loss (per epoch)
- Learning rate schedule
- Evaluation metrics (recall_at_k, coverage_at_k, ndcg_at_k)
- Dataset statistics (num_users, num_movies)
- Model size (total_parameters)
- Training time

**Tags:**
- `model_type` (retrieval, ranking, embedding, hybrid)
- `model_name` (two-tower, bert4rec, etc.)
- `framework` (pytorch)
- `status` (inactive → active when deployed)
- `dvc_pipeline` (true)

**Artifacts:**
- Trained model weights (PyTorch .pth)
- Metrics JSON file
- Training history
- Model configuration

## API Endpoints

### List Runs (with Filtering)

```bash
GET /api/models/runs?model_type=retrieval&limit=10

Response:
{
  "success": true,
  "data": [
    {
      "run_id": "abc123...",
      "run_name": "two-tower-20250104-143022",
      "model_type": "retrieval",
      "model_name": "two-tower",
      "metrics": {
        "recall_at_10": 0.234,
        "recall_at_50": 0.567,
        "training_time_seconds": 1234.5
      },
      "params": {
        "embedding_dim": 128,
        "learning_rate": 0.001
      }
    }
  ],
  "count": 10
}
```

### Get Run Details

```bash
GET /api/models/runs/{run_id}

Response:
{
  "success": true,
  "data": {
    "run_id": "abc123...",
    "run_name": "two-tower-20250104-143022",
    "start_time": 1704379822000,
    "end_time": 1704381056000,
    "metrics": {...},
    "params": {...},
    "tags": {...}
  }
}
```

### Dashboard Summary

```bash
GET /api/models/summary

Response:
{
  "success": true,
  "data": {
    "total_runs": 42,
    "counts_by_type": {
      "retrieval": 20,
      "ranking": 22
    },
    "counts_by_status": {
      "inactive": 40,
      "active": 2
    },
    "latest_models": [...]
  }
}
```

### Compare Models

```bash
POST /api/models/compare
Body: {"run_ids": ["run1", "run2", "run3"]}

Response:
{
  "success": true,
  "data": [
    {run1 data},
    {run2 data},
    {run3 data}
  ],
  "count": 3
}
```

## Files Created/Modified

### Backend Files

**New Files:**
1. `configs/mlflow.yaml` - MLflow configuration
2. `movie_genie/backend/app/services/mlflow_client.py` - MLflow Python client
3. `movie_genie/backend/app/api/models_routes.py` - Flask API routes

**Modified Files:**
1. `scripts/train_two_tower.py` - Added MLflow logging
2. `scripts/train_bert4rec.py` - Added MLflow logging
3. `dvc.yaml` - Added MLflow config dependency
4. `movie_genie/backend/app/api/__init__.py` - Registered models blueprint
5. `movie_genie/backend/app.py` - Enhanced startup logging
6. `.gitignore` - Added mlruns/, mlartifacts/, mlflow.db

### Frontend Files

**New Files:**
1. `frontend/src/types/models.ts` - TypeScript types
2. `frontend/src/services/modelMetricsService.ts` - API client
3. `frontend/src/hooks/useModelMetrics.ts` - React hooks
4. `frontend/src/components/ModelMetricsCard.tsx` - Card component
5. `frontend/src/components/ModelComparisonChart.tsx` - Comparison charts
6. `frontend/src/components/ModelMetricsDashboard.tsx` - Dashboard container
7. `frontend/src/pages/ModelMetrics.tsx` - Metrics page

**Modified Files:**
1. `frontend/src/App.tsx` - Added `/metrics` route
2. `frontend/src/pages/Index.tsx` - Added navigation button

## Usage

### Training with MLflow

```bash
# Train all models (MLflow tracks automatically)
dvc repro

# MLflow logs:
# - Hyperparameters
# - Training metrics (loss curves)
# - Evaluation metrics (recall, coverage)
# - Model artifacts
```

### View Experiments

**Option A: MLflow UI**

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

**Option B: Backend API**

```bash
curl http://localhost:5001/api/models/summary | jq
curl "http://localhost:5001/api/models/runs?limit=5" | jq
```

**Option C: Frontend Dashboard**

```bash
# Navigate to
http://localhost:8080/metrics
```

## Benefits

✅ **Automatic Experiment Tracking** - No manual logging needed
✅ **Model Lineage** - Track data → config → model → metrics
✅ **Easy Comparison** - Compare runs via UI or API
✅ **Version Control** - All models versioned in registry
✅ **Reproducibility** - Full parameter/metric history
✅ **Team Collaboration** - Shared experiment database
✅ **Production Ready** - Model promotion workflow
✅ **DVC Integration** - Seamless pipeline integration

## Important Notes

### Metric Naming

!!! warning "No Special Characters"
    MLflow doesn't allow `@` symbols in metric names.

    ✅ Correct: `recall_at_10`, `coverage_at_100`
    ❌ Wrong: `recall@10`, `coverage@100`

All metric names have been updated to use underscores.

## Testing

### 1. Test MLflow Logging

```bash
dvc repro two_tower_training
ls -la mlruns/  # Should see experiment directories
```

### 2. Test MLflow UI

```bash
mlflow ui --port 5000
# Visit http://localhost:5000 - should see runs
```

### 3. Test API

```bash
FLASK_PORT=5001 python scripts/start_server.py
curl http://localhost:5001/api/models/summary | jq
```

### 4. Test Frontend

```bash
cd movie_genie/frontend && npm run dev
# Visit http://localhost:8080/metrics
```

## Documentation

- **[MLflow Setup](setup.md)** - Complete setup guide
- **[How to Integrate MLflow](../how-to-guides/mlflow-integration.md)** - Step-by-step guide
- **[API Reference](../backend-frontend/api-reference.md)** - API documentation
- **[MLflow Official Docs](https://mlflow.org/docs/latest/)** - MLflow documentation

---

**Status**: Complete ✅
**Last Updated**: 2025-01-05
**Integration Level**: Production Ready 🚀
