# 🆘 Troubleshooting Guide

Common issues and solutions for Movie Genie development, deployment, and operation.

## 🎯 Quick Problem Solving

### Most Common Issues (90% of problems)

| Problem | Quick Fix | Section |
|---------|-----------|---------|
| **Frontend popup button grayed out** | Check backend is running on port 5001 | [Frontend Issues](#frontend-issues) |
| **"Module not found" errors** | Run `pip install -e .` in project root | [Installation Issues](#installation-issues) |
| **API not responding** | Check `curl http://127.0.0.1:5001/api/health` | [Backend Issues](#backend-issues) |
| **Models not loading** | Run `dvc repro` to train models | [ML Model Issues](#ml-model-issues) |
| **Database errors** | Delete and recreate: `rm movie_genie.db && dvc repro` | [Database Issues](#database-issues) |

---

## 📋 Documentation Sections

### [🧠 ML Model Issues](ml-model-issues.md)
**For**: Model training failures, prediction errors, memory issues
- Model loading and initialization problems
- Training pipeline failures
- Memory and GPU issues
- Performance optimization

### [🔧 Backend Issues](backend-issues.md)
**For**: Flask API problems, database connections, service errors
- API endpoint failures
- Database connectivity issues
- Service layer problems
- Authentication and permissions

### [🎨 Frontend Issues](frontend-issues.md)
**For**: React app problems, build failures, UI bugs
- Component rendering issues
- Build and deployment problems
- State management bugs
- Styling and responsive design

---

## 🔧 Installation Issues

### Python Environment Problems

#### "No module named 'movie_genie'"
```bash
# Solution: Install in development mode
pip install -e .

# If still failing, check virtual environment
which python
# Should point to .venv/bin/python

# Activate virtual environment if needed
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

#### "Permission denied" during installation
```bash
# Solution: Use virtual environment (preferred)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Or install for user only
pip install --user -e .
```

#### Python version conflicts
```bash
# Check Python version
python --version  # Should be 3.8+

# If wrong version, use specific Python
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Node.js and Frontend Issues

#### "npm install" failures
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf movie_genie/frontend/node_modules
cd movie_genie/frontend
npm install

# If still failing, check Node.js version
node --version  # Should be 16+
```

#### "EACCES" permission errors
```bash
# Fix npm permissions
npm config set prefix ~/.npm-global
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### DVC Issues

#### "DVC not initialized"
```bash
# Initialize DVC in project root
cd movie-genie
dvc init

# If git repository, use:
dvc init --no-scm
```

#### "Stage not found" errors
```bash
# Check dvc.yaml exists
ls -la dvc.yaml

# Verify stage names
dvc dag

# Run specific stage
dvc repro stage_name
```

---

## 🔌 Connection Issues

### Port Conflicts

#### "Port 5001 already in use"
```bash
# Find process using port
lsof -ti:5001

# Kill the process
lsof -ti:5001 | xargs kill -9

# Or use different port
FLASK_PORT=5002 python movie_genie/backend/app.py
```

#### Frontend can't reach backend
```bash
# Check backend is running
curl http://127.0.0.1:5001/api/health

# Check frontend environment
cat movie_genie/frontend/.env.development
# Should have: VITE_API_URL=http://127.0.0.1:5001/api

# Restart both services
# Terminal 1:
cd movie_genie/backend && python app.py

# Terminal 2:
cd movie_genie/frontend && npm run dev
```

### Network and Firewall Issues

#### API calls timing out
```bash
# Check firewall settings
# macOS:
sudo pfctl -s all

# Linux:
sudo ufw status

# Windows:
netsh advfirewall show allprofiles
```

#### CORS errors in browser
```python
# Backend app.py should have:
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
```

---

## 💾 Database Issues

### SQLite Database Problems

#### "Database is locked"
```bash
# Check if other processes are using database
lsof movie_genie/backend/movie_genie.db

# Kill processes if needed
pkill -f movie_genie

# Remove lock file if exists
rm movie_genie/backend/movie_genie.db-journal
```

#### "No such table" errors
```bash
# Recreate database
rm movie_genie/backend/movie_genie.db*
dvc repro setup_database

# Verify tables exist
sqlite3 movie_genie/backend/movie_genie.db ".tables"
```

#### Database corruption
```bash
# Check database integrity
sqlite3 movie_genie/backend/movie_genie.db "PRAGMA integrity_check;"

# If corrupted, restore from backup or recreate
rm movie_genie/backend/movie_genie.db
dvc repro data_processing setup_database
```

### Data Validation Issues

#### "No data found" errors
```bash
# Check data files exist
ls -la data/raw/ml-100k/
ls -la data/processed/

# If missing, run data processing
dvc repro data_processing

# Verify data quality
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/movies.parquet')
print(f'Movies: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
"
```

---

## 🚀 Performance Issues

### Slow Application Response

#### Backend API slow
```bash
# Check system resources
top
htop

# Monitor database queries
# Add to app.py:
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Frontend slow loading
```bash
# Check bundle size
cd movie_genie/frontend
npm run build
ls -lh dist/assets/

# Analyze bundle
npm install --save-dev webpack-bundle-analyzer
npm run build-analyze
```

### Memory Issues

#### "Out of memory" during training
```yaml
# Reduce batch sizes in configs/bert4rec_config.yaml
batch_size: 128  # Instead of 512

# Reduce model size
hidden_size: 64  # Instead of 128
num_layers: 2    # Instead of 4
```

#### High memory usage in production
```python
# Add memory monitoring to app.py
import psutil

@app.route('/api/health')
def health():
    memory = psutil.virtual_memory()
    return {
        'status': 'healthy',
        'memory_usage': f'{memory.percent}%',
        'available_memory': f'{memory.available / 1024**3:.1f}GB'
    }
```

---

## 🔍 Debugging Tools and Techniques

### Backend Debugging

#### Enable debug mode
```python
# In app.py
app.config['DEBUG'] = True
app.run(debug=True, host='127.0.0.1', port=5001)
```

#### Add request logging
```python
import logging
from flask import request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.before_request
def log_request_info():
    logger.info(f'{request.method} {request.url}')
```

#### Database query debugging
```python
# Add query logging
import sqlite3

def debug_query(query, params=None):
    print(f"SQL: {query}")
    if params:
        print(f"Params: {params}")
    return query
```

### Frontend Debugging

#### Browser developer tools
```javascript
// Add to MovieDataService
static async searchMovies(query) {
    console.log('🔍 Searching for:', query);

    try {
        const response = await fetch(`${API_URL}/search/semantic?q=${query}`);
        console.log('📡 API Response:', response.status);

        const data = await response.json();
        console.log('📄 Response Data:', data);

        return data;
    } catch (error) {
        console.error('❌ Search failed:', error);
        throw error;
    }
}
```

#### React component debugging
```tsx
// Add to components
import { useEffect } from 'react';

function MovieThumbnail({ movie }) {
    useEffect(() => {
        console.log('🎬 Rendering movie:', movie.title);
    }, [movie]);

    return (
        <div onClick={() => console.log('🖱️ Clicked movie:', movie.id)}>
            {/* component content */}
        </div>
    );
}
```

### ML Model Debugging

#### Test model loading
```python
# Create test script
python -c "
try:
    from movie_genie.ranking.bert4rec_model import BERT4RecReranker
    model = BERT4RecReranker('models/bert4rec/')
    print('✅ BERT4Rec loaded successfully')
except Exception as e:
    print(f'❌ BERT4Rec failed: {e}')
"
```

#### Check model files
```bash
# Verify model artifacts
ls -la models/
ls -la models/bert4rec/
ls -la models/two_tower/

# Check file sizes
du -sh models/*
```

---

## 📝 Logging and Monitoring

### Application Logs

#### Backend logging setup
```python
# Enhanced logging in app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

#### Monitor logs
```bash
# Real-time log monitoring
tail -f movie_genie/backend/logs/app.log

# Filter for errors
grep ERROR movie_genie/backend/logs/app.log

# Check access patterns
grep "GET\|POST" movie_genie/backend/logs/app.log | tail -20
```

### System Monitoring

#### Resource usage
```bash
# Monitor Python processes
ps aux | grep python

# Monitor memory usage
free -h

# Monitor disk usage
df -h
du -sh data/ models/ results/
```

#### Performance profiling
```bash
# Profile Python code
python -m cProfile -o profile.stats movie_genie/ranking/train_bert4rec.py

# Analyze profile
python -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(10)
"
```

---

## 🆘 Emergency Recovery

### Complete System Reset

#### Nuclear option - fresh start
```bash
# ⚠️ This will delete all generated data and models

# 1. Backup important files
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

# 5. Verify everything works
curl http://127.0.0.1:5001/api/health
```

### Partial Recovery

#### Reset just the models
```bash
# Remove and retrain models only
rm -rf models/
dvc repro train_bert4rec train_two_tower setup_semantic_search

# Test model loading
python -c "
from movie_genie.ranking.bert4rec_model import BERT4RecReranker
print('Models loaded successfully')
"
```

#### Reset just the database
```bash
# Remove and recreate database only
rm movie_genie/backend/movie_genie.db*
dvc repro setup_database

# Verify database
sqlite3 movie_genie/backend/movie_genie.db "SELECT COUNT(*) FROM movies;"
```

### Backup and Restore

#### Create backup
```bash
# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup database
cp movie_genie/backend/movie_genie.db $BACKUP_DIR/ 2>/dev/null || true

# Backup models
cp -r models/ $BACKUP_DIR/ 2>/dev/null || true

# Backup processed data
cp -r data/processed/ $BACKUP_DIR/ 2>/dev/null || true

echo "Backup created in $BACKUP_DIR"
EOF

chmod +x scripts/backup.sh
./scripts/backup.sh
```

#### Restore from backup
```bash
# List available backups
ls -la backups/

# Restore from specific backup
BACKUP_DATE="20240101_120000"
cp backups/$BACKUP_DATE/movie_genie.db movie_genie/backend/
cp -r backups/$BACKUP_DATE/models/ .
cp -r backups/$BACKUP_DATE/data/processed/ data/

# Verify restore
dvc status
```

---

## 📞 Getting Additional Help

### Self-Diagnostic Checklist

Before asking for help, run through this checklist:

- [ ] **Environment**: Virtual environment activated and dependencies installed
- [ ] **Services**: Backend running on port 5001, frontend on 5173
- [ ] **Data**: Database exists and contains data
- [ ] **Models**: ML models trained and loadable
- [ ] **Network**: API endpoints responding to curl tests
- [ ] **Logs**: Check application logs for specific error messages

### Useful Diagnostic Commands

```bash
# Complete system check
echo "=== System Check ==="
python --version
node --version
pip list | grep movie-genie
ls -la movie_genie/backend/movie_genie.db
curl -s http://127.0.0.1:5001/api/health | jq .

# DVC status
echo "=== DVC Status ==="
dvc status

# Model verification
echo "=== Model Check ==="
ls -la models/
python -c "import torch; print('PyTorch:', torch.__version__)"

# Database check
echo "=== Database Check ==="
sqlite3 movie_genie/backend/movie_genie.db "
SELECT 'Movies:' as table_name, COUNT(*) as count FROM movies
UNION ALL
SELECT 'Users:', COUNT(*) FROM users
UNION ALL
SELECT 'Ratings:', COUNT(*) FROM ratings;
"
```

### Reporting Issues

When reporting issues, include:

1. **Environment**: OS, Python version, Node.js version
2. **Steps to reproduce**: Exact commands that cause the issue
3. **Error messages**: Complete error output, not just snippets
4. **Logs**: Relevant log entries from application logs
5. **System state**: Output of diagnostic commands above

---

*This troubleshooting guide covers the most common issues you'll encounter. Remember: most problems have simple solutions, and systematic debugging usually reveals the root cause quickly.* 🆘