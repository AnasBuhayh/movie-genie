# Application Management

Complete guide for running and managing the Movie Genie application using the industry-standard application manager.

## Overview

Movie Genie now uses a unified application management script (`./scripts/app.sh`) that follows industry best practices for service orchestration. This separates concerns:

- **DVC Pipeline**: Data processing and ML training only
- **Application Manager**: Service lifecycle (start, stop, logs, etc.)

## Quick Start

```bash
# First time setup
./scripts/app.sh setup

# Run data pipeline + train models
./scripts/app.sh pipeline

# Start all services
./scripts/app.sh start

# View application
open http://127.0.0.1:5173
```

## Commands Reference

### Setup & Initialization

**`./scripts/app.sh setup`**

Install dependencies and initialize the project:
- Installs Python dependencies
- Installs frontend Node.js dependencies
- Creates necessary directories

```bash
./scripts/app.sh setup
```

### Data & Training Pipeline

**`./scripts/app.sh pipeline`**

Run the complete DVC pipeline:
- Data ingestion
- Data processing
- Model training
- Evaluation

```bash
./scripts/app.sh pipeline
```

**Equivalent to:**
```bash
dvc repro integrated_evaluation
```

### Build

**`./scripts/app.sh build`**

Build production frontend:
- Compiles React/TypeScript
- Optimizes assets
- Copies to backend static files

```bash
./scripts/app.sh build
```

### Service Management

**`./scripts/app.sh start`**

Start all services:
- Backend API (port 5001)
- Frontend dev server (port 5173)
- MLflow UI (port 5002)
- Documentation (port 8000)

```bash
./scripts/app.sh start
```

**`./scripts/app.sh dev`**

Start development services only (backend + frontend):

```bash
./scripts/app.sh dev
```

**`./scripts/app.sh stop`**

Stop all running services:

```bash
./scripts/app.sh stop
```

**`./scripts/app.sh restart`**

Restart all services:

```bash
./scripts/app.sh restart
```

### Monitoring

**`./scripts/app.sh status`**

Show status of all services:

```bash
./scripts/app.sh status
```

Output:
```
✅ Backend     - Running (PID: 12345, Port: 5001)
✅ Frontend    - Running (PID: 12346, Port: 5173)
✅ MLflow UI   - Running (PID: 12347, Port: 5002)
✅ Documentation - Running (PID: 12348, Port: 8000)
```

**`./scripts/app.sh logs [service]`**

View service logs:

```bash
# All logs
./scripts/app.sh logs all

# Specific service
./scripts/app.sh logs backend
./scripts/app.sh logs frontend
./scripts/app.sh logs mlflow
./scripts/app.sh logs docs
```

### Complete Workflow

**`./scripts/app.sh full`**

Run everything from start to finish:
1. Data pipeline
2. Model training
3. Frontend build
4. Start all services

```bash
./scripts/app.sh full
```

## Service Ports

| Service | Port | URL |
|---------|------|-----|
| **Frontend** (Dev) | 5173 | http://127.0.0.1:5173 |
| **Backend API** | 5001 | http://127.0.0.1:5001/api |
| **MLflow UI** | 5002 | http://127.0.0.1:5002 |
| **Documentation** | 8000 | http://127.0.0.1:8000 |

## Environment Variables

Customize ports via environment variables:

```bash
# Backend port
export FLASK_PORT=5001

# Frontend dev port
export VITE_PORT=5173

# MLflow UI port
export MLFLOW_PORT=5002

# Documentation port
export DOCS_PORT=8000

# Then start services
./scripts/app.sh start
```

## Common Workflows

### Development Workflow

```bash
# 1. Start development services
./scripts/app.sh dev

# 2. Make changes to code
# ...

# 3. View logs
./scripts/app.sh logs backend

# 4. Restart if needed
./scripts/app.sh restart
```

### Training New Models

```bash
# 1. Update model config
vim configs/two_tower.yaml

# 2. Run training
./scripts/app.sh pipeline

# 3. View results in MLflow
./scripts/app.sh start  # if not already running
open http://127.0.0.1:5002
```

### Production Build

```bash
# 1. Run full pipeline
./scripts/app.sh pipeline

# 2. Build frontend
./scripts/app.sh build

# 3. Start production backend
FLASK_ENV=production ./scripts/app.sh start
```

## File Locations

### PID Files
Service PIDs stored in `.pids/`:
```
.pids/
├── backend.pid
├── frontend.pid
├── mlflow.pid
└── docs.pid
```

### Log Files
Service logs in `logs/`:
```
logs/
├── backend.log
├── frontend.log
├── mlflow.log
└── docs.log
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i:5001

# Stop services
./scripts/app.sh stop

# Or kill specific process
kill $(lsof -ti:5001)
```

### Service Won't Start

```bash
# Check logs
./scripts/app.sh logs backend

# Check status
./scripts/app.sh status

# Try manual start
FLASK_PORT=5001 python scripts/start_server.py
```

### Services Running But Not Responding

```bash
# Restart all
./scripts/app.sh restart

# Or restart individually
./scripts/app.sh stop
# Wait a few seconds
./scripts/app.sh start
```

## Comparison with Old Workflow

### Old Way (DVC for Everything)

```bash
# ❌ DVC managed services (messy)
dvc repro backend_server
dvc repro mlflow_ui
dvc repro docs_server
```

### New Way (Proper Separation)

```bash
# ✅ DVC for data/ML only
dvc repro  # or ./scripts/app.sh pipeline

# ✅ Application manager for services
./scripts/app.sh start
```

## Advanced Usage

### Running Specific Services

Edit `scripts/app.sh` to add custom commands:

```bash
# Example: Start only MLflow
start_mlflow() {
    print_info "Starting MLflow UI..."
    mlflow ui --host 127.0.0.1 --port $MLFLOW_PORT
}
```

### Integration with CI/CD

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup
        run: ./scripts/app.sh setup
      - name: Run pipeline
        run: ./scripts/app.sh pipeline
      - name: Start services
        run: ./scripts/app.sh start
      - name: Run tests
        run: pytest
      - name: Stop services
        run: ./scripts/app.sh stop
```

### Docker Integration (Future)

The app manager script is designed to work seamlessly with Docker:

```bash
# Future docker-compose setup
docker-compose up  # internally uses ./scripts/app.sh
```

## Next Steps

- [DVC Pipeline Usage](dvc-pipeline.md)
- [Training Models](train-models.md)
- [MLflow Integration](mlflow-integration.md)

## Summary

The new application management system provides:

✅ **Clean Separation** - DVC for data/ML, app.sh for services
✅ **Industry Standard** - Similar to npm scripts, make, docker-compose
✅ **Unified Interface** - Single command for all operations
✅ **Proper Logging** - Centralized logs with easy access
✅ **Process Management** - PID tracking and graceful shutdown
✅ **Environment Control** - Configurable ports and settings

Use `./scripts/app.sh help` for complete command reference.
