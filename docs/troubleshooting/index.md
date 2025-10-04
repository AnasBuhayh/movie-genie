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

## Search Issues

### Semantic Search Returning Irrelevant Results

**Symptom**: Searching for "batman" returns "Zoolander" or other unrelated movies.

**Root Cause**: Using EmbeddingGemma-300M model which has poor semantic alignment.

**Solution**:
```bash
# 1. Verify current model
grep model_name configs/semantic_search.yaml

# Should be: sentence-transformers/all-MiniLM-L6-v2
# If not, update both configs:
vim configs/semantic_search.yaml  # Change model_name
vim configs/data.yaml              # Change embedding_model

# 2. Regenerate embeddings
dvc repro content_features

# 3. Restart backend
dvc repro backend_server

# 4. Test search quality
curl "http://127.0.0.1:5001/api/search/semantic?q=batman&k=5"
# Should return actual Batman movies
```

**Evidence of Problem**:
```python
# Test embedding quality
python3 << 'EOF'
from movie_genie.data.embeddings import TextEmbedder
import numpy as np

embedder = TextEmbedder("google/embeddinggemma-300M")  # BAD MODEL

batman_emb = embedder.embed_texts(["batman"])[0]
zoolander_emb = embedder.embed_texts(["zoolander comedy"])[0]

# Normalize
batman = batman_emb / np.linalg.norm(batman_emb)
zoolander = zoolander_emb / np.linalg.norm(zoolander_emb)

similarity = np.dot(batman, zoolander)
print(f"Similarity: {similarity:.4f}")  # Will be HIGH (0.85+) - WRONG!
EOF
```

---

### Search Engine Not Loading

**Symptom**: Backend logs show "Failed to initialize SemanticSearchEngine"

**Diagnostic Steps**:
```bash
# 1. Check if LLM dependencies installed
python3 -c "import torch; import transformers; print('✓ Dependencies OK')"

# If fails, install:
pip install -e ".[llm]"

# 2. Check embedding dimension
python3 << 'EOF'
import pandas as pd
import numpy as np
df = pd.read_parquet('data/processed/content_features.parquet')
if 'text_embedding' in df.columns:
    emb = np.array(df['text_embedding'].iloc[0])
    print(f"Dimension: {emb.shape}")  # Should be (384,)
else:
    print("❌ No text_embedding column found!")
EOF

# 3. Verify config consistency
grep model_name configs/semantic_search.yaml
grep embedding_model configs/data.yaml
# Both should match!
```

**Common Causes**:

1. **Missing torch/transformers**:
   ```bash
   pip install -e ".[llm]"
   ```

2. **Embedding dimension mismatch**:
   ```bash
   # The semantic_engine.py was hardcoded for 768-dim
   # Fixed to accept any dimension
   # If still broken, regenerate embeddings:
   dvc repro content_features
   ```

3. **Model download failed**:
   ```bash
   # Check cache
   ls ~/.cache/huggingface/hub/ | grep all-MiniLM

   # Force re-download if corrupted
   rm -rf ~/.cache/huggingface/hub/*all-MiniLM*
   dvc repro backend_server  # Will re-download
   ```

---

### Poster Images Not Showing

**Symptom**: Search results and movie thumbnails show placeholder text instead of images.

**Root Cause**: Backend not returning `poster_path` field or frontend not transforming it correctly.

**Diagnostic Steps**:
```bash
# 1. Check backend returns poster_path
curl "http://127.0.0.1:5001/api/search/semantic?q=batman&k=1" | jq '.data.movies[0].poster_path'

# Should return: "/xyz.jpg"
# If null, check search_service.py includes poster_path in results

# 2. Check frontend transformation
# Should be in movieDataService.ts:
#   poster_url: apiMovie.poster_path
#     ? `https://image.tmdb.org/t/p/w500${apiMovie.poster_path}`
#     : null
```

**Solution**:
```bash
# Already fixed in:
# - movie_genie/search/semantic_engine.py (line 146-147, 502-507)
# - movie_genie/backend/app/services/search_service.py (line 89)

# If broken, verify:
cat movie_genie/search/semantic_engine.py | grep poster_path
cat movie_genie/backend/app/services/search_service.py | grep poster_path
```

---

### Mock Data Showing Before Real Data

**Symptom**: Brief flash of "Popular Movie 1, 2, 3" before real data loads.

**Root Cause**: State not being cleared when new search/load starts.

**Solution**: Already fixed in `SearchResultsGrid.tsx` (lines 33-37):
```typescript
// Clear previous results immediately when new search starts
setSearchResults([]);
setIsLoading(true);
setTotalResults(0);
```

**Verify Fix**:
```bash
# Check the component clears state
grep -A 5 "Clear previous results" movie_genie/frontend/src/components/SearchResultsGrid.tsx
```

---

### Environment Variables Not Working (Vite)

**Symptom**: `VITE_USE_REAL_SEARCH=true` but still using mock data.

**Root Cause**: Using `process.env` instead of `import.meta.env` in Vite.

**Wrong**:
```typescript
const useRealData = process.env.VITE_USE_REAL_SEARCH === 'true';  // ❌ DOESN'T WORK
```

**Correct**:
```typescript
const useRealData = import.meta.env.VITE_USE_REAL_SEARCH !== 'false';  // ✅ WORKS
```

**Why `!== 'false'` instead of `=== 'true'`**:
- Vite env vars are `undefined` by default
- `undefined !== 'false'` = true (default enabled)
- `undefined === 'true'` = false (would disable by default)

**Verify**:
```bash
# Check movieDataService.ts uses correct pattern
grep "import.meta.env" movie_genie/frontend/src/services/movieDataService.ts
```

---

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