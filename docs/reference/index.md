# Reference Documentation

Technical reference materials and implementation details for Movie Genie.

## Quick Reference

<div class="grid cards" markdown>

-   :material-tools:{ .lg .middle } **Technology Stack**

    ---

    Complete overview of all technologies, frameworks, and tools used in Movie Genie.

    [:octicons-arrow-right-24: Technology Stack](technology-stack.md)

-   :material-folder-tree:{ .lg .middle } **Project Structure**

    ---

    Detailed file organization guide and architecture patterns.

    [:octicons-arrow-right-24: Project Structure](project-structure.md)

-   :material-code-tags:{ .lg .middle } **Coding Standards**

    ---

    Best practices, conventions, and style guidelines.

    [:octicons-arrow-right-24: Coding Standards](coding-standards.md)

-   :material-api:{ .lg .middle } **API Schema**

    ---

    Complete API specifications with request/response schemas.

    [:octicons-arrow-right-24: API Schema](api-schema.md)

</div>

## Architecture Quick Reference

### System Components
```
movie-genie/
├── Frontend (React + TypeScript)
├── Backend (Flask + Python)
├── ML Models (PyTorch)
├── Data Pipeline (DVC)
└── Documentation (MkDocs)
```

### Key Technologies
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Frontend | React | 18+ | User interface |
| Backend | Flask | 2.3+ | API server |
| ML Framework | PyTorch | 2.0+ | Model training |
| Data Pipeline | DVC | 3.0+ | Workflow management |
| Database | SQLite | 3.36+ | Data storage |
| Documentation | MkDocs | 1.6+ | Documentation site |

## API Quick Reference

### Base URL
```
http://127.0.0.1:5001/api
```

### Common Endpoints
| Endpoint | Method | Description | Example |
|----------|---------|-------------|---------|
| `/health` | GET | System status | `curl /api/health` |
| `/users/info` | GET | User information | `curl /api/users/info` |
| `/movies/popular` | GET | Popular movies | `curl /api/movies/popular?limit=10` |
| `/search/semantic` | GET | Semantic search | `curl "/api/search/semantic?q=action"` |
| `/recommendations/personalized` | GET | User recommendations | `curl "/api/recommendations/personalized?user_id=123"` |

### Response Format
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": { ... },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

## DVC Pipeline Quick Reference

### Pipeline Stages
```yaml
stages:
  data_processing:      # Clean and prepare data
  feature_engineering:  # Create ML features
  train_bert4rec:      # Train sequential model
  train_two_tower:     # Train collaborative model
  setup_semantic_search: # Setup content-based search
  setup_database:      # Create application database
  backend_server:      # Start application server
```

### Common Commands
```bash
dvc repro              # Run complete pipeline
dvc status             # Check what needs updating
dvc dag                # Show pipeline graph
dvc repro stage_name   # Run specific stage
```

## File Structure Quick Reference

### Core Directories
```
movie_genie/
├── backend/           # Flask application
│   ├── app.py        # Main server
│   ├── app/          # Application modules
│   └── templates/    # Static frontend files
├── frontend/         # React application
│   ├── src/          # Source code
│   └── dist/         # Built application
├── ranking/          # BERT4Rec model
├── retrieval/        # Two-Tower model
├── search/           # Semantic search
└── data/             # Data processing
```

### Configuration Files
```
configs/              # Model configurations
params.yaml          # Pipeline parameters
dvc.yaml             # Pipeline definition
mkdocs.yml           # Documentation config
package.json         # Frontend dependencies
pyproject.toml       # Python project config
```

## Model Quick Reference

### BERT4Rec
- **Type**: Sequential recommendation
- **Input**: User interaction sequences
- **Output**: Next-item probabilities
- **Training**: ~30 minutes
- **Best for**: Users with viewing history

### Two-Tower
- **Type**: Collaborative filtering
- **Input**: User and item features
- **Output**: User/item embeddings
- **Training**: ~10 minutes
- **Best for**: Large-scale recommendations

### Semantic Search
- **Type**: Content-based search
- **Input**: Natural language queries
- **Output**: Similarity rankings
- **Training**: Pre-trained
- **Best for**: Text-based discovery

## Development Quick Reference

### Backend Development
```bash
cd movie_genie/backend
python app.py                    # Start Flask server
pytest tests/                    # Run tests
curl http://127.0.0.1:5001/api/health  # Test API
```

### Frontend Development
```bash
cd movie_genie/frontend
npm install                      # Install dependencies
npm run dev                      # Start development server
npm test                         # Run tests
npm run build                    # Build for production
```

### Documentation
```bash
./scripts/docs.sh serve          # Start docs server
./scripts/docs.sh build          # Build documentation
mkdocs serve                     # Alternative docs command
```

## Environment Variables

### Backend (.env)
```bash
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_PORT=5001
DATABASE_URL=sqlite:///movie_genie.db
```

### Frontend (.env.development)
```bash
VITE_API_URL=http://127.0.0.1:5001/api
VITE_USE_REAL_POPULAR=true
VITE_USE_REAL_SEARCH=true
VITE_USE_REAL_RECOMMENDATIONS=true
```

## Performance Benchmarks

### Model Performance
- **BERT4Rec**: NDCG@10: 0.412, Recall@10: 0.278
- **Two-Tower**: NDCG@10: 0.385, Recall@10: 0.251
- **Semantic Search**: Sub-second query response

### System Performance
- **API Latency**: < 100ms average
- **Model Inference**: < 50ms per request
- **Database Queries**: < 10ms typical
- **Frontend Load**: < 2 seconds initial

## Common Patterns

### Error Handling
```python
try:
    result = some_operation()
    return {"success": True, "data": result}
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return {"success": False, "message": str(e)}
```

### API Response
```python
@app.route('/api/endpoint')
def endpoint():
    try:
        data = process_request()
        return jsonify({
            "success": True,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
```

### Service Layer
```python
class MovieService:
    def get_movies(self, **kwargs):
        # Business logic here
        return processed_data

    def _validate_input(self, data):
        # Input validation
        pass
```

This reference section provides quick access to all technical details and implementation patterns used throughout Movie Genie.