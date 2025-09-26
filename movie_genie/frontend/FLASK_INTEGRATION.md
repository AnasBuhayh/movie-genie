# Flask Integration Guide

This frontend has been prepared for seamless integration with a Flask backend serving your Movie Genie API.

## Setup Instructions

### 1. Install Dependencies
```bash
cd movie_genie/frontend
npm install
```

### 2. Environment Configuration
Copy the example environment file and customize:
```bash
cp .env.local.example .env.local
```

Edit `.env.local` to match your Flask server configuration.

### 3. Development Mode
Run the frontend in development mode (will proxy API calls to Flask):
```bash
npm run dev
```
Frontend will be available at `http://localhost:8080`

### 4. Production Build
Build the frontend for Flask integration:
```bash
npm run build
```

This will generate:
- HTML template in `../../templates/index.html`
- Static assets in `../../static/js/`, `../../static/css/`, etc.

## Flask Backend Requirements

Your Flask app should provide these API endpoints:

### Search Endpoints
- `GET /api/search/semantic?q=<query>` - Semantic movie search
- `GET /api/search?q=<query>` - Traditional movie search

### Movie Endpoints
- `GET /api/movies/<id>` - Get movie details
- `GET /api/movies/popular?limit=<n>` - Get popular movies

### Recommendation Endpoints
- `POST /api/recommendations/personalized` - Get personalized recommendations
  ```json
  {
    "user_id": "optional_user_id",
    "interaction_history": [...]
  }
  ```
- `GET /api/recommendations?movie_id=<id>&type=content_based` - Content-based recommendations

### User Feedback Endpoints
- `POST /api/feedback` - Submit user feedback
- `POST /api/rating` - Submit movie rating

### Response Formats

#### Movie Object
```json
{
  "movieId": 123,
  "title": "Movie Title",
  "overview": "Movie description...",
  "genres": ["Action", "Sci-Fi"],
  "release_date": "2023-01-01",
  "vote_average": 8.5,
  "vote_count": 1000,
  "poster_path": "/path/to/poster.jpg",
  "similarity_score": 0.85,
  "personalized_score": 0.92,
  "rank": 1
}
```

#### Search Response
```json
{
  "movies": [/* Movie objects */],
  "total": 100,
  "query": "search query",
  "search_type": "semantic"
}
```

#### Recommendation Response
```json
{
  "movies": [/* Movie objects */],
  "recommendation_type": "personalized",
  "user_context": {}
}
```

## Flask App Structure

### Basic Flask Setup
```python
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# API Routes
@app.route('/api/search/semantic')
def semantic_search():
    # Implement semantic search using your SemanticSearchEngine
    pass

@app.route('/api/recommendations/personalized', methods=['POST'])
def personalized_recommendations():
    # Implement personalized recommendations using BERT4Rec
    pass
```

## Integration with Your Models

The frontend is designed to work with your existing:
- **SemanticSearchEngine** for semantic movie search
- **BERT4Rec model** for personalized recommendations
- **Two-Tower model** for collaborative filtering
- **Content features** for content-based recommendations

## File Structure After Build

```
movie-genie/
├── templates/
│   └── index.html          # React app HTML template
├── static/
│   ├── js/                 # JavaScript bundles
│   ├── css/                # CSS files
│   ├── img/                # Images and assets
│   └── assets/             # Other static assets
└── movie_genie/
    └── frontend/           # Source code (for development)
```

## Development Workflow

1. **Frontend development**: Use `npm run dev` for hot reload
2. **API development**: Run Flask backend on `http://localhost:5000`
3. **Integration testing**: Use `npm run build` to test production build
4. **Deployment**: Flask serves the built frontend automatically

The frontend is now fully prepared for Flask integration with proper API calls, loading states, error handling, and production-ready builds!