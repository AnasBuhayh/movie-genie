# 📁 Project Structure

Complete guide to Movie Genie's file organization, directory structure, and code architecture.

## 🎯 Project Overview

Movie Genie follows a modular, scalable structure that separates concerns while maintaining clear relationships between components.

```
movie-genie/
├── 📊 Data & Models
├── 🏗️ Application Code
├── ⚙️ Configuration
├── 📚 Documentation
├── 🧪 Tests
└── 🚀 Deployment
```

---

## 📂 Complete Directory Structure

```
movie-genie/
├── configs/                          # ⚙️ Configuration files
│   ├── bert4rec_config.yaml          # BERT4Rec model configuration
│   ├── two_tower_config.yaml         # Two-Tower model configuration
│   ├── semantic_search.yaml          # Semantic search configuration
│   └── evaluation_config.yaml        # Model evaluation configuration
│
├── data/                             # 📊 Data storage
│   ├── raw/                          # Raw input data
│   │   └── ml-100k/                  # MovieLens 100K dataset
│   │       ├── u.data                # User ratings
│   │       ├── u.item                # Movie information
│   │       ├── u.user                # User demographics
│   │       └── u.genre               # Genre definitions
│   └── processed/                    # Processed data files
│       ├── movies.parquet            # Clean movie data
│       ├── ratings.parquet           # Clean rating data
│       ├── users.parquet             # User demographics
│       ├── content_features.parquet  # Movie content features
│       └── sequences_with_metadata.parquet # User sequences
│
├── docs/                             # 📚 Documentation
│   ├── README.md                     # Main documentation hub
│   ├── getting-started/              # Getting started guides
│   │   ├── README.md                 # Learning path overview
│   │   ├── quick-start.md            # 5-minute setup guide
│   │   ├── installation.md           # Detailed installation
│   │   ├── project-overview.md       # Architecture overview
│   │   └── commands-reference.md     # Complete command reference
│   ├── machine-learning/             # ML documentation
│   │   ├── README.md                 # ML models overview
│   │   ├── bert4rec.md               # Sequential recommendation
│   │   ├── two-tower.md              # Collaborative filtering
│   │   ├── semantic-search.md        # Content-based search
│   │   └── evaluation.md             # Performance evaluation
│   ├── data-pipeline/                # Data processing docs
│   │   ├── README.md                 # Pipeline overview
│   │   ├── dvc-workflows.md          # DVC pipeline management
│   │   ├── data-processing.md        # Data transformation
│   │   └── feature-engineering.md    # Feature creation
│   ├── backend-frontend/             # Full-stack architecture
│   │   ├── README.md                 # Architecture overview
│   │   ├── backend-integration.md    # Flask backend guide
│   │   ├── ml-integration.md         # ML to frontend guide
│   │   ├── api-reference.md          # Complete API docs
│   │   └── frontend-components.md    # React components
│   ├── deployment/                   # Deployment guides
│   ├── configuration/                # Configuration docs
│   ├── troubleshooting/              # Problem solving
│   │   └── README.md                 # Common issues guide
│   └── reference/                    # Technical reference
│       ├── technology-stack.md       # All technologies used
│       ├── project-structure.md      # This file
│       ├── coding-standards.md       # Best practices
│       └── changelog.md              # Project evolution
│
├── metrics/                          # 📈 Model performance metrics
│   ├── bert4rec_metrics.json         # BERT4Rec evaluation results
│   ├── two_tower_metrics.json        # Two-Tower evaluation results
│   └── comparison_report.json        # Model comparison
│
├── models/                           # 🧠 Trained ML models
│   ├── bert4rec/                     # BERT4Rec model artifacts
│   │   ├── bert4rec_model.pth        # Trained model weights
│   │   ├── config.json               # Model configuration
│   │   ├── tokenizer/                # Text tokenizer
│   │   └── training_log.json         # Training history
│   ├── two_tower/                    # Two-Tower model artifacts
│   │   ├── user_encoder.pth          # User embedding model
│   │   ├── item_encoder.pth          # Item embedding model
│   │   ├── config.json               # Model configuration
│   │   └── embeddings/               # Pre-computed embeddings
│   └── semantic_search/              # Semantic search models
│       ├── sentence_transformer/     # Pre-trained encoder
│       ├── movie_embeddings.npy      # Movie text embeddings
│       └── index.faiss               # Fast similarity search index
│
├── movie_genie/                      # 🏗️ Main application code
│   ├── __init__.py                   # Package initialization
│   │
│   ├── backend/                      # 🔧 Flask backend application
│   │   ├── __init__.py
│   │   ├── app.py                    # Main Flask application
│   │   ├── config.py                 # Backend configuration
│   │   ├── movie_genie.db            # SQLite database
│   │   ├── app/                      # Application modules
│   │   │   ├── __init__.py
│   │   │   ├── api/                  # API route handlers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── movies.py         # Movie endpoints
│   │   │   │   ├── search.py         # Search endpoints
│   │   │   │   ├── recommendations.py # Recommendation endpoints
│   │   │   │   └── users.py          # User endpoints
│   │   │   ├── services/             # Business logic layer
│   │   │   │   ├── __init__.py
│   │   │   │   ├── movie_service.py  # Movie data operations
│   │   │   │   ├── search_service.py # Search functionality
│   │   │   │   ├── recommendation_service.py # ML recommendations
│   │   │   │   └── user_service.py   # User operations
│   │   │   ├── models/               # Database models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── movie.py          # Movie data model
│   │   │   │   ├── rating.py         # Rating data model
│   │   │   │   └── user.py           # User data model
│   │   │   └── utils/                # Utility functions
│   │   │       ├── __init__.py
│   │   │       ├── ml_loader.py      # ML model loading
│   │   │       └── validators.py     # Input validation
│   │   ├── templates/                # Static files served by Flask
│   │   │   ├── index.html            # React app entry point
│   │   │   ├── favicon.ico           # Site icon
│   │   │   ├── robots.txt            # Search engine instructions
│   │   │   ├── placeholder.svg       # Placeholder image
│   │   │   └── assets/               # Built frontend assets
│   │   │       ├── index-*.js        # Bundled JavaScript
│   │   │       ├── index-*.css       # Bundled CSS
│   │   │       └── *.svg             # Optimized SVG assets
│   │   ├── tests/                    # Backend tests
│   │   │   ├── __init__.py
│   │   │   ├── test_api.py           # API endpoint tests
│   │   │   ├── test_services.py      # Service layer tests
│   │   │   └── test_models.py        # Database model tests
│   │   └── logs/                     # Application logs
│   │       ├── app.log               # Main application log
│   │       └── error.log             # Error-specific log
│   │
│   ├── frontend/                     # 🎨 React frontend application
│   │   ├── public/                   # Static assets
│   │   │   ├── favicon.ico           # Site icon
│   │   │   └── placeholder.svg       # Default movie poster
│   │   ├── src/                      # Source code
│   │   │   ├── components/           # React components
│   │   │   │   ├── UserSelectionModal.tsx # User ID selection
│   │   │   │   ├── MovieSearch.tsx   # Search interface
│   │   │   │   ├── SearchResultsGrid.tsx # Search results display
│   │   │   │   ├── MovieThumbnail.tsx # Movie card component
│   │   │   │   ├── RecommendationCarousel.tsx # Recommendations
│   │   │   │   └── MovieDetailsPanel.tsx # Movie details view
│   │   │   ├── services/             # Data access layer
│   │   │   │   ├── movieDataService.ts # Movie data operations
│   │   │   │   └── api.ts            # API client utilities
│   │   │   ├── lib/                  # Shared utilities
│   │   │   │   ├── api.ts            # API configuration
│   │   │   │   └── mockData.ts       # Mock data for development
│   │   │   ├── types/                # TypeScript type definitions
│   │   │   │   └── movie.ts          # Movie-related types
│   │   │   ├── styles/               # Global styles
│   │   │   │   └── index.css         # Main stylesheet
│   │   │   ├── App.tsx               # Main application component
│   │   │   ├── main.tsx              # Application entry point
│   │   │   └── vite-env.d.ts         # Vite type definitions
│   │   ├── dist/                     # Built application (production)
│   │   │   ├── index.html            # Built HTML
│   │   │   └── assets/               # Optimized assets
│   │   ├── node_modules/             # Node.js dependencies
│   │   ├── package.json              # Frontend dependencies
│   │   ├── package-lock.json         # Dependency lock file
│   │   ├── vite.config.ts            # Vite build configuration
│   │   ├── tailwind.config.js        # Tailwind CSS configuration
│   │   ├── tsconfig.json             # TypeScript configuration
│   │   ├── tsconfig.node.json        # Node-specific TypeScript config
│   │   ├── postcss.config.js         # PostCSS configuration
│   │   ├── .env.development          # Development environment vars
│   │   └── .env.production           # Production environment vars
│   │
│   ├── data/                         # 🔄 Data processing modules
│   │   ├── __init__.py
│   │   ├── content_features.py       # Movie content feature extraction
│   │   ├── sequential_processing.py  # User sequence generation
│   │   └── data_validation.py        # Data quality checks
│   │
│   ├── ranking/                      # 🎯 Sequential recommendation models
│   │   ├── __init__.py
│   │   ├── bert4rec_model.py         # BERT4Rec implementation
│   │   ├── train_bert4rec.py         # BERT4Rec training script
│   │   └── utils.py                  # Ranking utilities
│   │
│   ├── retrieval/                    # 🔍 Collaborative filtering models
│   │   ├── __init__.py
│   │   ├── two_tower_model.py        # Two-Tower implementation
│   │   ├── train_two_tower.py        # Two-Tower training script
│   │   └── embeddings.py             # Embedding utilities
│   │
│   ├── search/                       # 🔎 Semantic search modules
│   │   ├── __init__.py
│   │   ├── semantic_engine.py        # Semantic search engine
│   │   ├── setup_semantic_search.py  # Search setup script
│   │   └── text_processing.py        # Text preprocessing
│   │
│   └── evaluation/                   # 📊 Model evaluation modules
│       ├── __init__.py
│       ├── integrated_evaluation.py  # Cross-model evaluation
│       ├── metrics.py                # Evaluation metrics
│       └── benchmark.py              # Performance benchmarking
│
├── results/                          # 📈 Evaluation and experiment results
│   ├── model_comparison.json         # Model performance comparison
│   ├── ablation_studies/             # Feature importance studies
│   └── experiment_logs/              # Detailed experiment logs
│
├── scripts/                          # 🛠️ Utility and setup scripts
│   ├── setup_database.py             # Database initialization
│   ├── process_movielens.py          # MovieLens data processing
│   ├── imdb_featured_reviews.py      # IMDB review scraping
│   ├── generate_evaluation_report.py # Evaluation report generation
│   ├── test_full_pipeline.py         # End-to-end testing
│   └── backup.sh                     # System backup script
│
├── tests/                            # 🧪 Project-wide tests
│   ├── __init__.py
│   ├── test_ml_integration.py        # ML model integration tests
│   ├── test_api_integration.py       # API integration tests
│   ├── test_data_pipeline.py         # Data pipeline tests
│   └── fixtures/                     # Test data fixtures
│       ├── sample_movies.json        # Sample movie data
│       └── sample_ratings.json       # Sample rating data
│
├── .dvc/                             # 🔄 DVC configuration and cache
│   ├── config                        # DVC configuration
│   ├── cache/                        # DVC data cache
│   └── .gitignore                    # DVC gitignore rules
│
├── .venv/                            # 🐍 Python virtual environment
│   ├── bin/                          # Virtual environment binaries
│   ├── lib/                          # Python packages
│   └── pyvenv.cfg                    # Virtual environment config
│
├── dvc.yaml                          # 📋 DVC pipeline definition
├── dvc.lock                          # 🔒 DVC pipeline lock file
├── params.yaml                       # ⚙️ Pipeline parameters
├── .dvcignore                        # DVC ignore rules
├── .gitignore                        # Git ignore rules
├── pyproject.toml                    # Python project configuration
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview
├── LICENSE                           # License information
└── Dockerfile                        # Docker container definition
```

---

## 🏗️ Architecture Principles

### 1. Separation of Concerns
Each directory has a clear, single responsibility:
- **`movie_genie/backend/`**: HTTP API and business logic
- **`movie_genie/frontend/`**: User interface and interactions
- **`movie_genie/ranking/`**: Sequential recommendation models
- **`movie_genie/retrieval/`**: Collaborative filtering models
- **`movie_genie/search/`**: Content-based search

### 2. Modular Design
Components are loosely coupled and highly cohesive:
- **Service Layer**: Business logic separated from API routes
- **Component Architecture**: Reusable React components
- **Model Interfaces**: Standardized ML model interfaces

### 3. Configuration Management
All configuration is externalized:
- **`configs/`**: Model hyperparameters and settings
- **`params.yaml`**: Pipeline parameters
- **`.env` files**: Environment-specific variables

### 4. Data Flow Clarity
Clear data flow from raw to production:
```
data/raw/ → data/processed/ → models/ → movie_genie/backend/
```

---

## 📊 Key File Purposes

### Configuration Files

#### `dvc.yaml` - Pipeline Definition
```yaml
# Defines the complete ML pipeline
stages:
  data_processing:
    cmd: python scripts/process_movielens.py
    deps: [data/raw/ml-100k/]
    outs: [data/processed/]

  train_bert4rec:
    cmd: python movie_genie/ranking/train_bert4rec.py
    deps: [data/processed/sequences_with_metadata.parquet]
    outs: [models/bert4rec/]
```

#### `params.yaml` - Global Parameters
```yaml
# Shared parameters across all pipeline stages
data_processing:
  min_ratings_per_user: 20
  test_split_ratio: 0.2

bert4rec:
  hidden_size: 128
  num_layers: 4
  learning_rate: 0.001
```

#### `pyproject.toml` - Python Project Config
```toml
# Python package configuration and dependencies
[build-system]
requires = ["setuptools", "wheel"]

[project]
name = "movie-genie"
version = "1.0.0"
dependencies = [
    "flask>=2.3.0",
    "torch>=2.0.0",
    "transformers>=4.30.0"
]
```

### Core Application Files

#### `movie_genie/backend/app.py` - Flask Application
```python
# Main Flask application entry point
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Register API blueprints
from app.api import movies_bp, search_bp, recommendations_bp
app.register_blueprint(movies_bp, url_prefix='/api/movies')
app.register_blueprint(search_bp, url_prefix='/api/search')
app.register_blueprint(recommendations_bp, url_prefix='/api/recommendations')
```

#### `movie_genie/frontend/src/App.tsx` - React Application
```tsx
// Main React application component
import React, { useState } from 'react';
import { UserSelectionModal, MovieSearch, SearchResultsGrid } from './components';

export default function App() {
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);

  return (
    <div className="min-h-screen bg-gray-900">
      {!currentUser && <UserSelectionModal onUserSelect={setCurrentUser} />}
      {currentUser && (
        <>
          <MovieSearch />
          <SearchResultsGrid />
        </>
      )}
    </div>
  );
}
```

---

## 🔄 Data Flow Architecture

### 1. Data Processing Flow
```
data/raw/ml-100k/
├── u.data (ratings) ────┐
├── u.item (movies) ──────┼─→ scripts/process_movielens.py
└── u.user (users) ───────┘
                          │
                          ↓
                   data/processed/
                   ├── movies.parquet
                   ├── ratings.parquet
                   └── users.parquet
```

### 2. Model Training Flow
```
data/processed/ ──→ movie_genie/ranking/train_bert4rec.py ──→ models/bert4rec/
                └→ movie_genie/retrieval/train_two_tower.py ──→ models/two_tower/
                └→ movie_genie/search/setup_semantic_search.py ──→ models/semantic_search/
```

### 3. Application Flow
```
models/ ──→ movie_genie/backend/app.py ──→ API Endpoints
                                          │
                                          ↓
Frontend ←─── HTTP/JSON ←─── Flask Application
```

---

## 🎨 Frontend Structure Deep Dive

### Component Hierarchy
```
App.tsx
├── UserSelectionModal.tsx
├── MovieSearch.tsx
├── SearchResultsGrid.tsx
│   └── MovieThumbnail.tsx (multiple)
├── RecommendationCarousel.tsx
│   └── MovieThumbnail.tsx (multiple)
└── MovieDetailsPanel.tsx
    └── MovieThumbnail.tsx (similar movies)
```

### Service Layer
```
services/
├── movieDataService.ts    # Main data access layer
│   ├── getPopularMovies()
│   ├── searchMovies()
│   ├── getRecommendations()
│   └── getMovieDetails()
└── api.ts                 # Low-level API utilities
    ├── fetchAPI()
    └── API_ENDPOINTS
```

### Type Definitions
```typescript
// types/movie.ts
interface MovieData {
  id: string;
  title: string;
  genres: string[];
  poster_url: string | null;
  rating: number;
  // ... other fields
}

interface SearchResults {
  movies: MovieData[];
  total: number;
  query: string;
  hasRealData: boolean;
}
```

---

## 🔧 Backend Structure Deep Dive

### API Layer
```
app/api/
├── movies.py              # Movie-related endpoints
│   ├── GET /popular
│   ├── GET /{movie_id}
│   └── GET /similar/{movie_id}
├── search.py              # Search endpoints
│   ├── GET /semantic
│   └── GET /traditional
├── recommendations.py     # Recommendation endpoints
│   ├── GET /personalized
│   └── GET /similar/{movie_id}
└── users.py               # User endpoints
    └── GET /info
```

### Service Layer
```
app/services/
├── movie_service.py       # Movie business logic
│   ├── get_popular_movies()
│   ├── get_movie_by_id()
│   └── get_similar_movies()
├── recommendation_service.py # ML recommendation logic
│   ├── get_personalized_recommendations()
│   └── get_collaborative_recommendations()
├── search_service.py      # Search business logic
│   ├── semantic_search()
│   └── traditional_search()
└── user_service.py        # User business logic
    ├── get_user_info()
    └── get_user_stats()
```

### Model Layer
```
app/models/
├── movie.py               # Movie data model
├── rating.py              # Rating data model
└── user.py                # User data model
```

---

## 🧠 ML Module Structure

### Model Organization
```
movie_genie/
├── ranking/               # Sequential models
│   ├── bert4rec_model.py  # BERT4Rec implementation
│   └── train_bert4rec.py  # Training script
├── retrieval/             # Collaborative filtering
│   ├── two_tower_model.py # Two-Tower implementation
│   └── train_two_tower.py # Training script
└── search/                # Content-based search
    ├── semantic_engine.py # Search implementation
    └── setup_semantic_search.py # Setup script
```

### Model Interface Pattern
```python
# Standardized model interface
class BaseRecommender:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def predict(self, user_id: int, **kwargs) -> List[Dict]:
        """Return recommendations for user"""
        pass

    def load_model(self, path: str):
        """Load trained model from disk"""
        pass
```

---

## 📦 Deployment Structure

### Docker Structure
```
Dockerfile                 # Multi-stage Docker build
├── Frontend Build Stage   # Build React application
├── Python Dependencies    # Install Python packages
└── Production Stage       # Final runtime image
```

### Environment Configuration
```
.env files:
├── .env.development       # Development settings
├── .env.production        # Production settings
└── .env.example           # Template for environment variables
```

---

## 🧪 Testing Structure

### Test Organization
```
tests/
├── test_ml_integration.py    # ML model integration tests
├── test_api_integration.py   # API endpoint tests
├── test_data_pipeline.py     # Data processing tests
└── fixtures/                 # Test data
    ├── sample_movies.json
    └── sample_ratings.json

movie_genie/backend/tests/
├── test_api.py               # Backend API tests
├── test_services.py          # Service layer tests
└── test_models.py            # Database model tests

movie_genie/frontend/src/
└── __tests__/                # Frontend component tests
    ├── MovieThumbnail.test.tsx
    └── SearchResultsGrid.test.tsx
```

---

## 📈 Metrics and Monitoring

### Results Structure
```
results/
├── model_comparison.json     # Performance comparison
├── ablation_studies/         # Feature importance
│   ├── genre_importance.json
│   └── sequence_length_study.json
└── experiment_logs/          # Detailed logs
    ├── bert4rec_training.log
    └── two_tower_training.log

metrics/
├── bert4rec_metrics.json     # BERT4Rec evaluation
├── two_tower_metrics.json    # Two-Tower evaluation
└── comparison_report.json    # Model comparison
```

---

## 🎯 Development Workflow

### Adding New Features

#### 1. Backend API Endpoint
```bash
# 1. Add route handler
movie_genie/backend/app/api/new_feature.py

# 2. Add business logic
movie_genie/backend/app/services/new_feature_service.py

# 3. Add tests
movie_genie/backend/tests/test_new_feature.py
```

#### 2. Frontend Component
```bash
# 1. Add component
movie_genie/frontend/src/components/NewFeature.tsx

# 2. Add to main app
movie_genie/frontend/src/App.tsx

# 3. Add tests
movie_genie/frontend/src/__tests__/NewFeature.test.tsx
```

#### 3. ML Model
```bash
# 1. Add model implementation
movie_genie/new_model/model.py

# 2. Add training script
movie_genie/new_model/train.py

# 3. Update DVC pipeline
dvc.yaml

# 4. Add configuration
configs/new_model_config.yaml
```

---

*This project structure balances organization with simplicity, making it easy to navigate while maintaining clear separation of concerns. Each directory and file has a specific purpose and follows established conventions.* 📁