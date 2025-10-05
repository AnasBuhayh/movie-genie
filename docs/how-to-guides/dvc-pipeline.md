# How to Use the DVC Pipeline

Complete guide for running and managing the DVC pipeline in Movie Genie.

## Quick Start

```bash
# Run the complete pipeline
dvc repro

# Run specific stage
dvc repro two_tower_training

# Check pipeline status
dvc status

# Force re-run a stage
dvc repro --force backend_server
```

## Pipeline Stages

### Data Pipeline

1. **ingest** - Download and process MovieLens data
2. **sequential_processing** - Create user sequences
3. **content_features** - Extract TMDB metadata

### Model Training

4. **two_tower_training** - Train Two-Tower retrieval model
5. **bert4rec_training** - Train BERT4Rec ranking model

### Evaluation

6. **integrated_evaluation** - Evaluate complete system

### Web Application

7. **frontend_build** - Build React frontend
8. **backend_server** - Start Flask backend

## Common Commands

### Run Full Pipeline

```bash
# Everything from scratch
dvc repro

# Or use convenience script
python scripts/run_full_pipeline.py
```

### Run Specific Stages

```bash
# Data only
python scripts/run_full_pipeline.py --stage data

# Training only
python scripts/run_full_pipeline.py --stage training

# Web app only
python scripts/run_full_pipeline.py --web-only
```

### Check Status

```bash
# DVC status
dvc status

# Detailed status
python scripts/run_full_pipeline.py --status
```

### Force Re-run

```bash
# Force specific stage
dvc repro --force two_tower_training

# Force all training
dvc repro --force two_tower_training bert4rec_training
```

## Understanding DVC Output

### Stage is Up-to-Date

```bash
Stage 'two_tower_training' didn't change, skipping
```
**Meaning**: Dependencies haven't changed, output already exists

### Stage Will Run

```bash
Running stage 'two_tower_training':
> .venv/bin/python scripts/train_two_tower.py ...
```
**Meaning**: Dependencies changed or output missing

### Stage Failed

```bash
ERROR: failed to reproduce 'two_tower_training'
```
**Meaning**: Command returned non-zero exit code

## Modifying the Pipeline

### Adding a New Stage

Edit `dvc.yaml`:

```yaml
  your_new_stage:
    cmd: >
      .venv/bin/python scripts/your_script.py
      --config configs/your_config.yaml
    deps:
      - scripts/your_script.py
      - configs/your_config.yaml
      - data/processed/input.parquet
    outs:
      - models/your_model:
          cache: false
    metrics:
      - models/your_model/metrics.json:
          cache: false
```

Then run:

```bash
dvc repro your_new_stage
```

### Updating Dependencies

When you add new dependencies to a stage:

```yaml
deps:
  - scripts/train.py
  - configs/config.yaml
  - data/new_dependency.parquet  # Added this
```

DVC will automatically detect the change and re-run the stage.

### Changing Parameters

Parameters are tracked in the `cmd` section. Changing any part of the command will trigger a re-run:

```yaml
cmd: >
  .venv/bin/python scripts/train.py
  --learning-rate 0.001  # Changed from 0.0001
```

## Pipeline Dependencies

### Stage Dependencies

```
data → training → evaluation → web
```

Each stage depends on previous stages' outputs.

### Automatic Invalidation

When you change an upstream stage, downstream stages automatically re-run:

```bash
# Changed data processing
dvc repro sequential_processing

# This will trigger:
# 1. sequential_processing (changed)
# 2. two_tower_training (depends on sequences)
# 3. bert4rec_training (depends on sequences)
# 4. integrated_evaluation (depends on both models)
```

## Working with Metrics

### View Metrics

```bash
# Show all metrics
dvc metrics show

# Show specific metric file
dvc metrics show models/two_tower/metrics.json

# Compare metrics across runs
dvc metrics diff
```

### Metrics Format

DVC expects JSON format:

```json
{
  "recall_at_10": 0.234,
  "recall_at_50": 0.456,
  "training_time_seconds": 1234.5
}
```

## Troubleshooting

### Pipeline Stuck

```bash
# Check if process is actually running
ps aux | grep python

# Check CPU usage
top

# If truly stuck, kill and restart
pkill -f "train_two_tower"
dvc repro --force two_tower_training
```

### Dependencies Not Detected

```bash
# Check lock file
cat dvc.lock

# Force update
dvc repro --force stage_name
```

### Cache Issues

```bash
# Clear DVC cache
dvc cache clear

# Run from scratch
dvc repro --force
```

### Backend Server Won't Stop

```bash
# Use stop script
./scripts/stop_server.sh

# Or manually
pkill -f "flask run"
lsof -ti:5001 | xargs kill -9
```

## Best Practices

### 1. Always Check Status First

```bash
dvc status
```

### 2. Use Force Sparingly

Only use `--force` when necessary - it invalidates caching.

### 3. Commit dvc.lock

```bash
git add dvc.lock
git commit -m "Updated pipeline"
```

### 4. Test Stages Individually

Before running full pipeline:

```bash
# Test one stage first
dvc repro two_tower_training

# If successful, run full pipeline
dvc repro
```

## Advanced Usage

### Running with Different Configs

```bash
# Modify config
vim configs/two_tower.yaml

# Run with changes
dvc repro two_tower_training
```

### Parallel Execution

DVC automatically runs independent stages in parallel:

```bash
# Both models train in parallel
dvc repro two_tower_training bert4rec_training
```

### Debugging Pipeline

```bash
# Verbose output
dvc repro --verbose

# Dry run (show what would run)
dvc repro --dry
```

## Integration with Git

### Committing Changes

```bash
# After modifying pipeline
git add dvc.yaml dvc.lock
git commit -m "Updated training config"

# DVC outputs are gitignored
# Only configs and lock file are tracked
```

### Reproducing from Git

```bash
# Clone repo
git clone <repo>

# Reproduce pipeline
dvc repro

# All outputs regenerated from tracked configs
```

## Next Steps

- [Adding a New Model](add-new-model.md)
- [MLflow Integration](mlflow-integration.md)
- [DVC Documentation](https://dvc.org/doc)
