# Movie Genie DVC Pipeline

Your Movie Genie project now includes a complete DVC pipeline that manages the entire ML workflow and web application deployment.

## 🚀 Quick Start

### Run Complete Pipeline
```bash
# Run everything: data processing, training, evaluation, and web app
dvc repro

# Or use the convenience script
python scripts/run_full_pipeline.py

# 🛑 IMPORTANT: To stop the server and cleanup processes
./scripts/stop_server.sh
```

### Run Web Application Only
```bash
# Skip training, just run the web app
python scripts/run_full_pipeline.py --web-only

# Or run individual stages
dvc repro frontend_build
dvc repro backend_server
```

## 📋 Pipeline Stages

### 1. Data Processing
```bash
dvc repro ingest                    # Ingest raw MovieLens data
dvc repro sequential_processing     # Create user sequences
dvc repro content_features          # Extract TMDB movie features
```

### 2. Model Training
```bash
dvc repro two_tower_training        # Train Two-Tower model
dvc repro bert4rec_training         # Train BERT4Rec model
```

### 3. Evaluation
```bash
dvc repro integrated_evaluation     # Evaluate complete system
```

### 4. Web Application
```bash
dvc repro frontend_build           # Build React frontend
dvc repro backend_server           # Start Flask backend
```

## 🎯 DVC Commands

### Check Pipeline Status
```bash
dvc status                         # Show pipeline status
python scripts/run_full_pipeline.py --status  # Detailed status
```

### Run Specific Stages
```bash
# Data pipeline only
python scripts/run_full_pipeline.py --stage data

# Training only
python scripts/run_full_pipeline.py --stage training

# Web application only
python scripts/run_full_pipeline.py --stage web
```

### Force Re-run Stages
```bash
dvc repro --force backend_server   # Force restart backend
dvc repro --force frontend_build   # Force rebuild frontend
```

## 📊 Pipeline Flow

```
Data Sources → Processing → Training → Evaluation → Web App
     ↓             ↓           ↓           ↓          ↓
┌─────────┐ ┌─────────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐
│MovieLens│ │User Sequences│ │Two-Tower│ │Integrated│ │Flask API│
│  TMDB   │ │Content Feats │ │BERT4Rec │ │Evaluation│ │React Web│
└─────────┘ └─────────────┘ └────────┘ └──────────┘ └─────────┘
```

## 🌐 Web Application

Once the pipeline runs, your web application will be available at:

- **Frontend**: http://127.0.0.1:5000
- **API**: http://127.0.0.1:5000/api
- **Health Check**: http://127.0.0.1:5000/api/health

### API Endpoints
- `GET /api/movies/popular` - Popular movies
- `GET /api/search/semantic?q=query` - Semantic search
- `POST /api/recommendations/personalized` - Personalized recommendations
- `POST /api/feedback` - Submit user feedback

## 🔧 Configuration

### Environment Variables
```bash
# Backend configuration (movie_genie/backend/.env)
FLASK_ENV=development
FLASK_PORT=5000
SECRET_KEY=your-secret-key

# Frontend configuration (movie_genie/frontend/.env.local)
VITE_API_URL=http://localhost:5000/api
```

### Model Paths
All paths are automatically configured through DVC:
- **Two-Tower Model**: `models/two_tower/`
- **BERT4Rec Model**: `models/bert4rec/`
- **Movie Data**: `data/processed/content_features.parquet`
- **User Sequences**: `data/processed/sequences_with_metadata.parquet`

## 🛠️ Troubleshooting

### Common Issues

**Backend won't start:**
```bash
# Check if models are trained
dvc status
ls models/bert4rec/ models/two_tower/

# Retrain if needed
dvc repro two_tower_training bert4rec_training
```

**Frontend not loading:**
```bash
# Rebuild frontend
dvc repro --force frontend_build

# Check static files
ls movie_genie/backend/static/
ls movie_genie/backend/templates/
```

**DVC stage fails:**
```bash
# Check detailed logs
dvc repro --verbose STAGE_NAME

# Force clean rebuild
dvc repro --force STAGE_NAME
```

## 📈 Monitoring

### View Metrics
```bash
# Training metrics
cat metrics/two_tower_metrics.json
cat metrics/bert4rec_metrics.json

# Evaluation results
cat results/integrated_system_evaluation.json
```

### Backend Logs
The Flask backend provides detailed logs including:
- Model loading status
- API request/response logs
- Error messages and debugging info

## 🎬 Production Deployment

For production deployment:

1. **Set production environment:**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-production-secret
   ```

2. **Use production WSGI server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 movie_genie.backend.app:app
   ```

3. **Set up reverse proxy** (nginx, Apache, etc.)

4. **Use production database** (PostgreSQL, MySQL, etc.)

Your complete Movie Genie system is now fully integrated with DVC for reproducible ML workflows and seamless web application deployment! 🎉