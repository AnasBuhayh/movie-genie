# 📋 Commands Reference

Complete reference of all commands you'll need for developing, running, and maintaining Movie Genie.

## 🚀 Quick Start Commands

### **Essential Commands**
```bash
# 1. Install the project
pip install -e .

# 2. Run the complete pipeline
dvc repro

# 3. Access the application
# Open browser to: http://127.0.0.1:5001
```

---

## 📦 DVC Pipeline Commands

### **Run Complete Pipeline**
```bash
# Run all stages from data processing to deployment
dvc repro

# Run specific stage
dvc repro stage_name

# Run stages up to a specific point
dvc repro --downstream stage_name
```

### **Check Pipeline Status**
```bash
# Check what needs to be reproduced
dvc status

# Show pipeline graph
dvc dag

# Show detailed pipeline information
dvc pipeline show --ascii
```

### **Manage Data and Models**
```bash
# Add data to DVC tracking
dvc add data/raw/movies.csv

# Download data from remote storage
dvc pull

# Upload data to remote storage
dvc push

# Check data status
dvc data status
```

---

## 🔄 Data Processing Commands

### **Database Operations**
```bash
# Remove existing database and test fresh setup
rm -f movie_genie.db*
python test_db.py

# Initialize database with fresh data
python scripts/setup_database.py

# Backup database
cp movie_genie/backend/movie_genie.db movie_genie_backup.db
```

### **Data Scraping and Collection**
```bash
# Run IMDB reviews scraper
python scripts/imdb_featured_reviews.py \
  --links-csv data/raw/ml-100k/links.csv \
  --limit 25 \
  --out data/raw/imdb-reviews/ml-100k_reviews.csv \
  --lang en \
  --min-delay 0.05 \
  --max-delay 0.1 \
  --checkpoint data/raw/imdb-reviews/ml-100k_checkpoint.json \
  --filter-by-movies

# Process MovieLens data
python scripts/process_movielens.py \
  --input data/raw/ml-100k/ \
  --output data/processed/
```

### **Feature Engineering**
```bash
# Generate content features
python movie_genie/data/content_features.py \
  --input data/raw/ \
  --output data/processed/content_features.parquet

# Create user sequences
python movie_genie/data/sequential_processing.py \
  --input data/processed/ \
  --output data/processed/sequences_with_metadata.parquet
```

---

## 🧠 ML Model Commands

### **Model Training**
```bash
# Train BERT4Rec model
python movie_genie/ranking/train_bert4rec.py \
  --config configs/bert4rec_config.yaml \
  --data data/processed/sequences_with_metadata.parquet \
  --output models/bert4rec/

# Train Two-Tower model
python movie_genie/retrieval/train_two_tower.py \
  --config configs/two_tower_config.yaml \
  --data data/processed/ \
  --output models/two_tower/

# Setup semantic search
python movie_genie/search/setup_semantic_search.py \
  --config configs/semantic_search.yaml \
  --data data/processed/content_features.parquet
```

### **Model Evaluation**
```bash
# Run integrated evaluation
python movie_genie/evaluation/integrated_evaluation.py \
  --config configs/evaluation_config.yaml \
  --models models/ \
  --data data/processed/ \
  --output results/

# Generate evaluation report
python scripts/generate_evaluation_report.py \
  --results results/ \
  --output reports/model_comparison.html
```

### **Model Testing**
```bash
# Test BERT4Rec predictions
python -c "
from movie_genie.ranking.bert4rec_model import BERT4RecReranker
reranker = BERT4RecReranker('models/bert4rec/')
print(reranker.predict(user_id=123, num_recommendations=10))
"

# Test semantic search
python -c "
from movie_genie.search.semantic_engine import SemanticSearchEngine
engine = SemanticSearchEngine('configs/semantic_search.yaml')
print(engine.search('action movies with robots', k=10))
"
```

---

## 🌐 Backend & Frontend Commands

### **Backend Development**
```bash
# Start backend server (development)
cd movie_genie/backend
python app.py

# Start backend with specific port
FLASK_PORT=5001 python app.py

# Start with production settings
FLASK_ENV=production python app.py

# Test API endpoints
curl http://127.0.0.1:5001/api/health
curl http://127.0.0.1:5001/api/users/info
curl "http://127.0.0.1:5001/api/search/semantic?q=action%20movies"
```

### **Frontend Development**
```bash
# Install frontend dependencies
cd movie_genie/frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run type checking
npm run type-check

# Run linting
npm run lint
```

### **Full-Stack Development**
```bash
# Run both backend and frontend in development
# Terminal 1: Backend
cd movie_genie/backend && python app.py

# Terminal 2: Frontend
cd movie_genie/frontend && npm run dev

# Or use DVC pipeline for integrated setup
dvc repro backend_server
```

---

## 🧪 Testing Commands

### **Backend Testing**
```bash
# Run all backend tests
pytest movie_genie/backend/tests/

# Run specific test file
pytest movie_genie/backend/tests/test_api.py

# Run with coverage
pytest --cov=movie_genie movie_genie/backend/tests/

# Test specific API endpoint
pytest movie_genie/backend/tests/test_api.py::test_user_info_endpoint
```

### **Frontend Testing**
```bash
# Run frontend tests
cd movie_genie/frontend
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run end-to-end tests
npm run test:e2e
```

### **Integration Testing**
```bash
# Test full pipeline
python scripts/test_full_pipeline.py

# Test ML model integration
python tests/test_ml_integration.py

# Test API integration
python tests/test_api_integration.py
```

---

## 🐳 Deployment Commands

### **Docker Deployment**
```bash
# Build Docker image
docker build -t movie-genie .

# Run Docker container
docker run -p 5001:5001 movie-genie

# Run with Docker Compose
docker-compose up -d

# Check logs
docker logs movie-genie
docker-compose logs -f
```

### **Production Deployment**
```bash
# Install production dependencies
pip install -r requirements.txt --no-dev

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 movie_genie.backend.app:app

# Run with systemd service
sudo systemctl start movie-genie
sudo systemctl enable movie-genie
sudo systemctl status movie-genie
```

---

## 🔧 Development Utilities

### **Code Quality**
```bash
# Format Python code
black movie_genie/
isort movie_genie/

# Lint Python code
flake8 movie_genie/
pylint movie_genie/

# Type checking
mypy movie_genie/

# Format frontend code
cd movie_genie/frontend
npm run format
npm run lint:fix
```

### **Environment Management**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Update dependencies
pip-tools compile requirements.in
pip-sync requirements.txt

# Check environment
pip list
pip check
```

### **Data Management**
```bash
# Clean generated files
rm -rf models/ results/ metrics/
dvc cache dir --unset

# Reset to clean state
dvc clean
git clean -fdx

# Check data integrity
dvc data status
dvc check-ignore data/
```

---

## 🔍 Debugging Commands

### **Backend Debugging**
```bash
# Run backend with debug mode
FLASK_ENV=development FLASK_DEBUG=1 python movie_genie/backend/app.py

# Check database content
sqlite3 movie_genie/backend/movie_genie.db
.tables
SELECT COUNT(*) FROM movies;
SELECT COUNT(*) FROM ratings;

# Monitor API requests
tail -f movie_genie/backend/logs/app.log
```

### **Frontend Debugging**
```bash
# Start with verbose logging
cd movie_genie/frontend
VITE_LOG_LEVEL=debug npm run dev

# Check bundle size
npm run build-analyze

# Debug network requests
# Open browser dev tools → Network tab
```

### **ML Model Debugging**
```bash
# Check model files
ls -la models/bert4rec/
ls -la models/two_tower/

# Verify model loading
python -c "
import torch
model = torch.load('models/bert4rec/bert4rec_model.pth')
print(f'Model type: {type(model)}')
print(f'Model keys: {model.keys() if isinstance(model, dict) else \"Not a dict\"}')"

# Test data loading
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/content_features.parquet')
print(f'Data shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')"
```

---

## 📊 Monitoring Commands

### **System Health**
```bash
# Check API health
curl http://127.0.0.1:5001/api/health

# Monitor system resources
htop
nvidia-smi  # If using GPU

# Check application logs
tail -f movie_genie/backend/logs/app.log
journalctl -u movie-genie -f
```

### **Performance Monitoring**
```bash
# Profile Python code
python -m cProfile -o profile.stats movie_genie/ranking/train_bert4rec.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Monitor memory usage
python -m memory_profiler movie_genie/ranking/train_bert4rec.py

# Check disk usage
du -sh data/ models/ results/
df -h
```

---

## 🆘 Emergency Commands

### **Quick Recovery**
```bash
# Reset everything to working state
git stash
git checkout main
dvc checkout
dvc repro

# Restore from backup
cp movie_genie_backup.db movie_genie/backend/movie_genie.db

# Force rebuild everything
dvc clean --all
dvc repro --force
```

### **Clean Installation**
```bash
# Remove all generated files
rm -rf .venv/ models/ data/processed/ results/
rm -f movie_genie/backend/movie_genie.db

# Fresh installation
python -m venv .venv
source .venv/bin/activate
pip install -e .
dvc repro
```

---

*Keep this reference handy during development. Most daily tasks can be accomplished with these commands!* 📚