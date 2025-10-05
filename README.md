# Movie Genie

An AI-powered movie recommendation system with semantic search, demonstrating modern ML engineering and full-stack development best practices.

## Overview

Movie Genie is a production-ready movie recommendation system that showcases:

- **Advanced ML Models**: BERT4Rec (sequential), Two-Tower (collaborative), Semantic Search (NLP-powered)
- **Modern Full-Stack**: React TypeScript frontend with Flask Python backend
- **MLOps Pipeline**: DVC-managed reproducible workflows
- **Real Data**: MovieLens dataset with TMDB metadata (9,742 movies, 610 users)
- **Semantic Search**: Natural language movie discovery using sentence-transformers

Perfect for learning modern ML engineering, full-stack development, and production MLOps practices.

## Key Features

- **Semantic Search**: Find movies using natural language ("action movies with robots")
- **Personalized Recommendations**: BERT4Rec sequential recommendations based on viewing history
- **User Analytics**: Genre preferences, watched movies, historical interest
- **Modern UI**: React + TypeScript + Tailwind CSS with real movie posters
- **Fast**: Sub-second search with 384-dimensional embeddings
- **Reproducible**: Complete DVC pipeline for data processing and model training

## Quick Start

### Production Deployment (Docker)

**Prerequisites**: Docker 20.10+ and Docker Compose 2.0+

```bash
# 1. Clone and setup
git clone https://github.com/AnasBuhayh/movie-genie.git
cd movie-genie

# 2. Install dependencies and build artifacts with DVC
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[llm]"

# IMPORTANT: Run dvc repro from within the activated virtual environment
dvc repro  # ~15 minutes: downloads models, processes data, trains ML models

# 3. Configure environment
cp .env.example .env
# Edit .env and set SECRET_KEY for production

# 4. Build and deploy with Docker
docker-compose build
docker-compose up -d

# Access the application
# Frontend: http://localhost:8080
# Backend API: http://localhost:5001/api
# MLflow UI: http://localhost:5002
# Documentation: http://localhost:8000
```

**See [Docker Deployment Guide](docs/how-to-guides/docker-deployment.md) for production setup, monitoring, and troubleshooting.**

### Local Development

**Prerequisites**: Python 3.9+, Node.js 16+, Git, 8GB+ RAM, 5GB+ disk space

```bash
# 1. Clone and setup
git clone https://github.com/AnasBuhayh/movie-genie.git
cd movie-genie
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[llm]"

# 2. Build artifacts with DVC (one time)
dvc repro  # ~15 minutes first run

# 3. Run development servers
# Terminal 1: Frontend
cd movie_genie/frontend
npm install
npm run dev  # http://localhost:5173

# Terminal 2: Backend
python scripts/start_server.py  # http://localhost:5001
```

**Development Benefits**: Hot reloading, faster iteration, no Docker overhead

## Key Commands

### Production (Docker)
```bash
# Build artifacts (required first)
dvc repro

# Start all services
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Check service health
docker-compose ps
curl http://localhost:5001/api/health
```

### Development (Local)
```bash
# Frontend (hot reload)
cd movie_genie/frontend
npm run dev

# Backend
python scripts/start_server.py

# MLflow UI
mlflow ui --host 127.0.0.1 --port 5002

# Documentation
mkdocs serve
```

### DVC Pipeline
```bash
# Build all artifacts (required before Docker deployment)
dvc repro

# Run specific stages
dvc repro content_features    # Process data + generate embeddings
dvc repro bert4rec_training   # Train BERT4Rec model
dvc repro frontend_build      # Build React frontend

# Check pipeline status
dvc status                    # See what's out of date
dvc dag                       # View pipeline graph
```

### Testing & Debugging
```bash
# API health check
curl http://127.0.0.1:5001/api/health

# Test semantic search
curl "http://127.0.0.1:5001/api/search/semantic?q=action%20movies&k=5"

# Test user endpoints
curl http://127.0.0.1:5001/api/users/1/watched

# Search engine status
curl http://127.0.0.1:5001/api/search/status
```

## Architecture

### System Overview
```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  React Frontend │ ◄─────► │  Flask Backend   │ ◄─────► │   ML Models     │
│  (TypeScript)   │  REST   │  (Python)        │         │  (PyTorch)      │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │ Data Pipeline    │
                            │ (DVC + Parquet)  │
                            └──────────────────┘
```

### Technology Stack
- **Frontend**: React 18 + TypeScript + Tailwind CSS + Vite + shadcn/ui
- **Backend**: Flask + Python 3.9+ + CORS
- **ML Models**: PyTorch + Transformers + sentence-transformers
- **Data**: Parquet (not SQLite) for efficient ML data access
- **Pipeline**: DVC for reproducible workflows
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)

## Machine Learning Models

### Model Comparison

| Model | Type | Purpose | Dimension | Training Time |
|-------|------|---------|-----------|---------------|
| **BERT4Rec** | Sequential Transformer | Personalized recommendations | 256 | ~30 min |
| **Two-Tower** | Neural Collaborative | User-item matching | 128 | ~10 min |
| **Semantic Search** | Sentence Transformer | Natural language search | 384 | Pre-trained ✅ |

### Semantic Search Details

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Why this model**: Better semantic quality than EmbeddingGemma-300M
- **Embedding dimension**: 384 (efficient and fast)
- **Search method**: Cosine similarity on normalized embeddings
- **Reranking**: BERT4Rec for personalized results
- **Performance**: Sub-second query processing on 9,742 movies

**Model Selection Story**:
We initially used EmbeddingGemma-300M but discovered it had poor semantic alignment (searching "batman" ranked "Zoolander" higher than Batman movies!). Switching to all-MiniLM-L6-v2 dramatically improved search quality.

## Dataset

**Source**: MovieLens + TMDB metadata

**Statistics**:
- **Movies**: 9,742 with rich metadata
- **Users**: 610 with rating histories
- **Ratings**: 100,000+ interactions
- **Genres**: 19 unique genres
- **Posters**: 98.7% coverage from TMDB
- **Embeddings**: 384-dimensional semantic vectors for all movies

**Data Processing**:
- Feature engineering with TMDB data
- Text embedding generation (title + overview + keywords)
- Sequential data preparation for BERT4Rec
- Genre parsing and normalization

## API Endpoints

### Search
- `GET /search/semantic` - Natural language search with ML embeddings
- `GET /search/traditional` - Title-based search (fallback)
- `GET /search/status` - Search engine health check

### Users
- `GET /users/info` - User statistics
- `GET /users/{id}/profile` - Full interaction history
- `GET /users/{id}/watched` - Recently watched movies
- `GET /users/{id}/historical-interest` - Genre-based recommendations

### Movies
- `GET /movies/popular` - Popular movies by vote count
- `GET /movies/{id}` - Movie details with metadata

### Recommendations
- `POST /recommendations/personalized` - BERT4Rec personalized recommendations

**Full API documentation**: See `docs/backend-frontend/api-reference.md`

## Documentation

### View Documentation

Start the documentation server for easy navigation:

```bash
# Serve interactive documentation
./scripts/serve_docs.sh
# Or: mkdocs serve

# Access at: http://127.0.0.1:8000
```

**Documentation includes:**

- **[How-To Guides](http://127.0.0.1:8000/how-to-guides/)**: Step-by-step implementation guides
  - [Adding a New Model](http://127.0.0.1:8000/how-to-guides/add-new-model/)
  - [MLflow Integration](http://127.0.0.1:8000/how-to-guides/mlflow-integration/)
  - [Serving Recommendations](http://127.0.0.1:8000/how-to-guides/serve-recommendations/)
  - [Training Models](http://127.0.0.1:8000/how-to-guides/train-models/)
  - [DVC Pipeline Usage](http://127.0.0.1:8000/how-to-guides/dvc-pipeline/)
  - [Docker Deployment](http://127.0.0.1:8000/how-to-guides/docker-deployment/)

- **[MLflow Documentation](http://127.0.0.1:8000/mlflow/)**: Experiment tracking and metrics
  - [Setup Guide](http://127.0.0.1:8000/mlflow/setup/)
  - [Integration Summary](http://127.0.0.1:8000/mlflow/integration-summary/)
  - [Frontend Dashboard](http://127.0.0.1:8000/mlflow/dashboard-implementation/)

- **[Machine Learning](http://127.0.0.1:8000/machine-learning/)**: Model architectures
  - [BERT4Rec](http://127.0.0.1:8000/machine-learning/bert4rec/)
  - [Two-Tower](http://127.0.0.1:8000/machine-learning/two_tower/)
  - [Semantic Search](http://127.0.0.1:8000/machine-learning/semantic-search/)

- **[API Reference](http://127.0.0.1:8000/backend-frontend/api-reference/)**: Complete endpoint docs
- **[Troubleshooting](http://127.0.0.1:8000/troubleshooting/)**: Common issues and solutions

## Configuration

### Key Configuration Files

**Semantic Search** (`configs/semantic_search.yaml`):
```yaml
model_name: "sentence-transformers/all-MiniLM-L6-v2"
normalize_vectors: true
reranker:
  enabled: true
  type: "bert4rec"
  semantic_weight: 0.6
  personalization_weight: 0.4
```

**Data Processing** (`configs/data.yaml`):
```yaml
data_sources:
  movielens:
    dataset_size: "ml-latest-small"
processing:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

**Frontend** (`movie_genie/frontend/.env.development`):
```bash
VITE_API_URL=http://127.0.0.1:5001/api
VITE_USE_REAL_SEARCH=true
```

See [Configuration Reference](docs/reference/configuration.md) for all parameters.

## Performance Metrics

### Model Performance
- **BERT4Rec NDCG@10**: 0.412
- **BERT4Rec Recall@10**: 0.278
- **Semantic Search Accuracy**: High relevance for natural language queries
- **Search Speed**: <100ms for typical queries

### System Performance
- **API Response**: <100ms for most endpoints
- **Model Inference**: <50ms for recommendations
- **Search Engine Init**: ~5 seconds (loads 9,742 embeddings)
- **Frontend Load**: <2 seconds initial load

## Development

### Project Structure
```
movie-genie/
├── movie_genie/
│   ├── backend/           # Flask API
│   │   ├── app/          # API routes and services
│   │   └── app.py        # Application entry point
│   ├── frontend/         # React TypeScript app
│   │   └── src/          # Components and services
│   ├── data/             # Data loaders and processors
│   ├── ranking/          # BERT4Rec model
│   ├── retrieval/        # Two-Tower model
│   └── search/           # Semantic search engine
├── configs/              # YAML configurations
├── data/                 # Dataset storage
│   ├── raw/             # Original MovieLens data
│   └── processed/       # Parquet files with embeddings
├── models/              # Trained model weights
├── docs/                # Documentation
└── dvc.yaml            # Pipeline definition
```

### Adding New Features

See [Common Workflows](docs/getting-started/common-workflows.md) for detailed guides on:
- Changing embedding models
- Adding new API endpoints
- Debugging search issues
- Regenerating data with DVC

## Troubleshooting

### Common Issues

**Search returns irrelevant results**:
```bash
# Check semantic search model
grep model_name configs/semantic_search.yaml
# Should be: sentence-transformers/all-MiniLM-L6-v2

# Regenerate embeddings if wrong
dvc repro content_features
```

**Backend fails to start**:
```bash
# Check dependencies
pip install -e ".[llm]"

# Check search engine status
curl http://127.0.0.1:5001/api/search/status
```

**Environment variables not working**:
```typescript
// In Vite, use import.meta.env (NOT process.env)
const apiUrl = import.meta.env.VITE_API_URL;
```

See [Troubleshooting Guide](docs/troubleshooting/index.md) for complete solutions.

## Contributing

This is a portfolio/learning project, but contributions are welcome:

1. **Documentation**: Improve clarity or add examples
2. **Features**: Enhance UI or add new recommendation algorithms
3. **Tests**: Add unit or integration tests
4. **Performance**: Optimize search or model inference

Please open an issue first to discuss major changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MovieLens**: For the recommendation dataset
- **TMDB**: For movie metadata and poster images
- **Hugging Face**: For transformer models and sentence-transformers
- **PyTorch**: For the deep learning framework
- **DVC**: For data and model versioning
- **React + Vite**: For the modern frontend experience

---

**Movie Genie** demonstrates production-ready ML engineering with semantic search, personalized recommendations, and modern full-stack development. Perfect for learning recommendation systems, NLP, and MLOps workflows.

**Live Demo**: [Add your deployment URL]
**Documentation**: [Full docs](docs/)
**Author**: Anas Buhayh
