# Troubleshooting Guide

Quick solutions for common Movie Genie issues and debugging workflows.

## Quick Problem Solver

Most issues fall into these categories. Click the relevant section for immediate solutions:

<div class="grid cards" markdown>

-   :material-alert-circle:{ .lg .middle } **Installation Issues**

    ---

    Module not found, permission errors, Python environment problems

    **Quick Fixes**:
    - `pip install -e .`
    - Check virtual environment activation
    - Verify Python 3.8+ version

-   :material-server-network:{ .lg .middle } **Server Connection**

    ---

    API not responding, CORS errors, port conflicts

    **Quick Fixes**:
    - Check `curl http://127.0.0.1:5001/api/health`
    - Verify backend is running
    - Kill process on port 5001 if needed

-   :material-brain:{ .lg .middle } **ML Model Issues**

    ---

    Models not loading, training failures, memory errors

    **Quick Fixes**:
    - Run `dvc repro` to retrain models
    - Check model files exist in `models/`
    - Reduce batch sizes in `params.yaml`

-   :material-database:{ .lg .middle } **Database Problems**

    ---

    Database locked, missing tables, data corruption

    **Quick Fixes**:
    - Delete and recreate: `rm movie_genie.db && dvc repro setup_database`
    - Check file permissions
    - Verify data integrity

</div>

## Emergency Quick Fixes

### 🚨 Most Common Issues (90% of problems)

| Problem | Quick Command | More Info |
|---------|---------------|-----------|
| **Frontend button grayed out** | Check backend: `curl http://127.0.0.1:5001/api/health` | [Connection Issues](#connection-issues) |
| **Module not found** | `pip install -e .` | [Installation](#installation-issues) |
| **Port in use** | `lsof -ti:5001 \| xargs kill -9` | [Port Conflicts](#port-conflicts) |
| **Models not loading** | `dvc repro` | [ML Issues](#ml-model-issues) |
| **Database errors** | `rm movie_genie.db && dvc repro setup_database` | [Database Issues](#database-issues) |

## Diagnostic Commands

Run these commands to quickly identify issues:

```bash
# System check
echo "=== Movie Genie System Status ==="
python --version
node --version
pip list | grep movie-genie

# Backend check
echo "=== Backend Status ==="
curl -s http://127.0.0.1:5001/api/health || echo "Backend not running"

# Database check
echo "=== Database Status ==="
sqlite3 movie_genie/backend/movie_genie.db "SELECT COUNT(*) as movies FROM movies;" 2>/dev/null || echo "Database not found"

# Model check
echo "=== Models Status ==="
ls -la models/ 2>/dev/null || echo "Models directory not found"

# DVC check
echo "=== DVC Status ==="
dvc status
```

## Common Issues by Category

### Installation Issues

#### "No module named 'movie_genie'"
```bash
# Solution: Install in development mode
pip install -e .

# Verify installation
pip list | grep movie-genie
```

#### Permission denied
```bash
# Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

#### Python version conflicts
```bash
# Check version
python --version  # Should be 3.8+

# If wrong version
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Connection Issues

#### Backend not responding
```bash
# Check if backend is running
curl http://127.0.0.1:5001/api/health

# If not, start backend
cd movie_genie/backend
python app.py
```

#### Port conflicts
```bash
# Find what's using port 5001
lsof -ti:5001

# Kill the process
lsof -ti:5001 | xargs kill -9

# Start backend again
cd movie_genie/backend && python app.py
```

#### CORS errors
```python
# Check Flask app has CORS enabled
# In movie_genie/backend/app.py
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # This line should be present
```

### ML Model Issues

#### Models not loading
```bash
# Check if models exist
ls -la models/

# If missing, train models
dvc repro train_bert4rec train_two_tower setup_semantic_search

# Verify model files
ls -la models/bert4rec/
ls -la models/two_tower/
```

#### Out of memory during training
```yaml
# Edit params.yaml - reduce batch sizes
bert4rec:
  batch_size: 128  # Instead of 256
  hidden_size: 64  # Instead of 128

two_tower:
  batch_size: 256  # Instead of 512
```

#### Model inference errors
```python
# Test model loading
python -c "
try:
    from movie_genie.ranking.bert4rec_model import BERT4RecReranker
    model = BERT4RecReranker('models/bert4rec/')
    print('✅ BERT4Rec loaded successfully')
except Exception as e:
    print(f'❌ BERT4Rec failed: {e}')
"
```

### Database Issues

#### Database locked
```bash
# Check for lock files
ls -la movie_genie/backend/movie_genie.db*

# Remove lock and restart
rm movie_genie/backend/movie_genie.db-journal
pkill -f movie_genie
```

#### Missing tables
```bash
# Recreate database
rm movie_genie/backend/movie_genie.db
dvc repro setup_database

# Verify tables
sqlite3 movie_genie/backend/movie_genie.db ".tables"
```

#### Data integrity issues
```bash
# Check database integrity
sqlite3 movie_genie/backend/movie_genie.db "PRAGMA integrity_check;"

# If corrupted, recreate
rm movie_genie/backend/movie_genie.db
dvc repro data_processing setup_database
```

## Frontend Issues

#### Build failures
```bash
# Clear npm cache
cd movie_genie/frontend
npm cache clean --force
rm -rf node_modules
npm install
```

#### Environment variables not working
```bash
# Check environment file
cat movie_genie/frontend/.env.development

# Should contain:
# VITE_API_URL=http://127.0.0.1:5001/api
# VITE_USE_REAL_POPULAR=true
```

## Performance Issues

### Slow application response
```bash
# Check system resources
top
htop  # If available

# Monitor API response times
time curl http://127.0.0.1:5001/api/movies/popular
```

### High memory usage
```bash
# Monitor memory usage
ps aux | grep python

# Check model memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

## DVC Pipeline Issues

#### Stage not found
```bash
# Check pipeline definition
cat dvc.yaml

# Show available stages
dvc dag
```

#### Dependencies not found
```bash
# Check file paths
ls -la data/raw/ml-100k/

# Verify dependencies
dvc status --verbose
```

#### Pipeline stuck
```bash
# Force reproduction
dvc repro --force

# Clean and restart
dvc clean
dvc repro
```

## Emergency Recovery

### Complete system reset
```bash
# ⚠️ This will delete all generated data and models

# 1. Backup important files (optional)
cp movie_genie/backend/movie_genie.db backup_db.db 2>/dev/null || true

# 2. Clean everything
rm -rf .venv/ models/ data/processed/ results/
rm -f movie_genie/backend/movie_genie.db*

# 3. Fresh installation
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 4. Rebuild everything
dvc repro
```

### Partial recovery
```bash
# Reset just models
rm -rf models/
dvc repro train_bert4rec train_two_tower setup_semantic_search

# Reset just database
rm movie_genie/backend/movie_genie.db*
dvc repro setup_database
```

## Getting Help

Before asking for help, please:

1. **Run the diagnostic commands** above
2. **Check the error messages** - they usually contain the solution
3. **Try the emergency recovery** if nothing else works
4. **Include system information** when reporting issues:

```bash
# Collect system info for bug reports
echo "System: $(uname -a)"
echo "Python: $(python --version)"
echo "Node: $(node --version 2>/dev/null || echo 'Not installed')"
echo "DVC: $(dvc version 2>/dev/null || echo 'Not installed')"
echo "Working directory: $(pwd)"
echo "Virtual env: $VIRTUAL_ENV"
```

Most issues have simple solutions. Start with the quick fixes and work your way through the diagnostic steps systematically.