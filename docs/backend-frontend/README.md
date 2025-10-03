# 🌐 Backend & Frontend Architecture

Complete guide to Movie Genie's full-stack architecture, from Flask API design to React frontend integration.

## 🎯 Architecture Overview

Movie Genie follows a modern full-stack architecture with clear separation of concerns:

```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   Frontend      │◄──────────────►│    Backend      │
│   (React TS)    │    JSON API     │    (Flask)      │
│                 │                 │                 │
│ • Components    │                 │ • API Endpoints │
│ • State Mgmt    │                 │ • Business Logic│
│ • Data Services │                 │ • ML Integration│
└─────────────────┘                 └─────────────────┘
         │                                   │
         │                          ┌─────────────────┐
         │                          │   Database      │
         │                          │   (SQLite)      │
         │                          │                 │
         │                          │ • Movies        │
         │                          │ • Ratings       │
         │                          │ • Users         │
         │                          └─────────────────┘
         │
         │                          ┌─────────────────┐
         └──────────────────────────│   ML Models     │
                  Via API           │   (PyTorch)     │
                                    │                 │
                                    │ • BERT4Rec      │
                                    │ • Two-Tower     │
                                    │ • Semantic      │
                                    └─────────────────┘
```

## 📋 Documentation Sections

### [🔧 Backend Integration](backend-integration.md)
**Perfect for**: Understanding Flask API development and ML model integration
- Flask application structure and blueprints
- Database models and queries
- Service layer design patterns
- Error handling and logging

### [🤖 ML Integration](ml-integration.md)
**Perfect for**: Connecting ML models to web applications
- Model loading and inference
- API endpoint design for ML services
- Real/mock data switching for development
- Performance optimization and caching

### [📚 API Reference](api-reference.md)
**Perfect for**: Complete API documentation and testing
- All endpoint specifications
- Request/response formats
- Authentication and error codes
- Example API calls and responses

### [🎨 Frontend Components](frontend-components.md)
**Perfect for**: React development and UI architecture
- Component hierarchy and props
- State management patterns
- Styling with Tailwind CSS
- TypeScript integration

---

## 🏗️ Backend Architecture

### Flask Application Structure
```
movie_genie/backend/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── movie_genie.db        # SQLite database
├── app/
│   ├── __init__.py
│   ├── api/              # API blueprints
│   │   ├── movies.py     # Movie endpoints
│   │   ├── search.py     # Search endpoints
│   │   ├── recommendations.py # Recommendation endpoints
│   │   └── users.py      # User endpoints
│   ├── services/         # Business logic
│   │   ├── movie_service.py
│   │   ├── search_service.py
│   │   ├── recommendation_service.py
│   │   └── user_service.py
│   ├── models/           # Database models
│   │   ├── movie.py
│   │   ├── rating.py
│   │   └── user.py
│   └── utils/            # Utility functions
│       ├── ml_loader.py
│       └── validators.py
├── templates/            # Static assets served by Flask
│   ├── index.html        # React app entry point
│   ├── favicon.ico
│   └── assets/           # Built frontend assets
└── logs/                 # Application logs
```

### API Design Principles
- **RESTful Routes**: Standard HTTP methods and resource naming
- **JSON Responses**: Consistent response format with success/error handling
- **Service Layer**: Business logic separated from route handlers
- **Error Handling**: Comprehensive error messages and status codes
- **Validation**: Input validation and sanitization
- **Documentation**: Clear endpoint documentation with examples

### Database Schema
```sql
-- Core entities
CREATE TABLE movies (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT,
    release_date DATE,
    overview TEXT,
    poster_path TEXT,
    vote_average REAL
);

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    occupation TEXT
);

CREATE TABLE ratings (
    user_id INTEGER,
    movie_id INTEGER,
    rating INTEGER,
    timestamp INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (movie_id) REFERENCES movies(id)
);
```

---

## 🎨 Frontend Architecture

### React Application Structure
```
movie_genie/frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── UserSelectionModal.tsx
│   │   ├── MovieSearch.tsx
│   │   ├── SearchResultsGrid.tsx
│   │   ├── MovieThumbnail.tsx
│   │   ├── RecommendationCarousel.tsx
│   │   └── MovieDetailsPanel.tsx
│   ├── services/         # Data access layer
│   │   ├── movieDataService.ts
│   │   └── api.ts
│   ├── lib/              # Utilities and helpers
│   │   ├── api.ts
│   │   └── mockData.ts
│   ├── types/            # TypeScript type definitions
│   │   └── movie.ts
│   ├── styles/           # CSS and styling
│   │   └── index.css
│   ├── App.tsx           # Main application component
│   ├── main.tsx          # Application entry point
│   └── vite-env.d.ts     # Vite type definitions
├── public/               # Static assets
├── dist/                 # Built application (production)
├── package.json          # Dependencies and scripts
├── vite.config.ts        # Vite configuration
├── tailwind.config.js    # Tailwind CSS configuration
└── tsconfig.json         # TypeScript configuration
```

### Component Architecture
```typescript
// Component hierarchy
App
├── UserSelectionModal     // User authentication
├── MovieSearch           // Search interface
├── SearchResultsGrid     // Search results display
│   └── MovieThumbnail[]  // Individual movie cards
├── RecommendationCarousel // Personalized suggestions
│   └── MovieThumbnail[]  // Movie cards in carousel
└── MovieDetailsPanel     // Detailed movie information
    └── MovieThumbnail[]  // Similar movies
```

### State Management
```typescript
// Application state structure
interface AppState {
  currentUser: UserInfo | null;
  searchQuery: string;
  searchResults: MovieData[];
  isSearching: boolean;
  selectedMovie: MovieData | null;
  popularMovies: MovieData[];
  recommendations: MovieData[];
  loading: {
    popular: boolean;
    search: boolean;
    recommendations: boolean;
  };
  errors: {
    [key: string]: string | null;
  };
}
```

---

## 🔄 Data Flow Patterns

### API Request Flow
```typescript
// 1. User Action (e.g., search)
const handleSearch = async (query: string) => {
  setIsSearching(true);

  try {
    // 2. Service Layer Call
    const results = await MovieDataService.searchMovies(query);

    // 3. State Update
    setSearchResults(results.movies);
    setIsSearching(false);
  } catch (error) {
    // 4. Error Handling
    setError('Search failed');
    setIsSearching(false);
  }
};

// Service Layer Implementation
export class MovieDataService {
  static async searchMovies(query: string): Promise<SearchResults> {
    // 5. API Call
    const response = await fetch(`${API_URL}/search/semantic?q=${query}`);
    const data = await response.json();

    // 6. Data Transformation
    if (data.success) {
      return {
        movies: data.data.movies.map(this.transformApiMovie),
        total: data.data.total,
        query
      };
    }

    throw new Error(data.message);
  }
}
```

### Real/Mock Data Switching
```typescript
// Environment-controlled data sources
const DATA_SOURCE_CONFIG = {
  popular: import.meta.env.VITE_USE_REAL_POPULAR === 'true',
  search: import.meta.env.VITE_USE_REAL_SEARCH === 'true',
  recommendations: import.meta.env.VITE_USE_REAL_RECOMMENDATIONS === 'true',
  movieDetails: import.meta.env.VITE_USE_REAL_MOVIE_DETAILS === 'true'
};

// Service method with fallback
static async getPopularMovies(limit: number = 20): Promise<MovieData[]> {
  if (DATA_SOURCE_CONFIG.popular) {
    try {
      // Try real API first
      const response = await fetch(`${API_URL}/movies/popular?limit=${limit}`);
      const data = await response.json();

      if (data.success) {
        return data.data.movies.map(this.transformApiMovie);
      }
    } catch (error) {
      console.warn('Real API failed, falling back to mock data:', error);
    }
  }

  // Fallback to mock data
  return this.getMockPopularMovies(limit);
}
```

---

## 🔧 Development Workflow

### Backend Development
```bash
# Start backend development server
cd movie_genie/backend
python app.py

# Test API endpoints
curl http://127.0.0.1:5001/api/health
curl http://127.0.0.1:5001/api/movies/popular

# Run backend tests
pytest movie_genie/backend/tests/

# Format code
black movie_genie/backend/
```

### Frontend Development
```bash
# Start frontend development server
cd movie_genie/frontend
npm run dev

# Open browser to http://localhost:5173
# Hot reload enabled for development

# Run frontend tests
npm test

# Build for production
npm run build
```

### Full-Stack Development
```bash
# Terminal 1: Backend
cd movie_genie/backend && python app.py

# Terminal 2: Frontend
cd movie_genie/frontend && npm run dev

# Or use DVC pipeline for integrated setup
dvc repro backend_server
```

---

## 🚀 Production Deployment

### Build Process
```bash
# 1. Build frontend
cd movie_genie/frontend
npm run build

# 2. Copy built assets to backend
cp -r dist/* ../backend/templates/

# 3. Start production server
cd ../backend
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### Docker Deployment
```dockerfile
# Multi-stage Docker build
FROM node:16 AS frontend-build
WORKDIR /app/frontend
COPY movie_genie/frontend/package*.json ./
RUN npm install
COPY movie_genie/frontend/ ./
RUN npm run build

FROM python:3.9-slim AS backend
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY movie_genie/backend/ ./backend/
COPY --from=frontend-build /app/frontend/dist/ ./backend/templates/
EXPOSE 5001
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "backend.app:app"]
```

---

## 📊 Performance Considerations

### Backend Optimization
- **Database Indexing**: Optimize queries with proper indexes
- **Caching**: Use Redis for frequently accessed data
- **Connection Pooling**: Manage database connections efficiently
- **Async Processing**: Use Celery for background tasks

### Frontend Optimization
- **Code Splitting**: Load components on demand
- **Image Optimization**: Compress and lazy-load images
- **Bundle Analysis**: Monitor and optimize bundle size
- **Service Workers**: Cache API responses and assets

### API Performance
```python
# Example caching implementation
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_popular_movies(limit: int) -> List[Dict]:
    """Cache popular movies for 5 minutes"""
    return movie_service.get_popular_movies(limit)

# Rate limiting
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/search')
@limiter.limit("10 per minute")
def search_movies():
    pass
```

---

## 🔍 Testing Strategy

### Backend Testing
```python
# Unit tests for services
def test_movie_service_get_popular():
    service = MovieService()
    movies = service.get_popular_movies(limit=10)
    assert len(movies) <= 10
    assert all('title' in movie for movie in movies)

# Integration tests for API
def test_api_popular_movies(client):
    response = client.get('/api/movies/popular?limit=5')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert len(data['data']) <= 5
```

### Frontend Testing
```typescript
// Component tests
import { render, screen } from '@testing-library/react';
import { MovieThumbnail } from './MovieThumbnail';

test('renders movie title', () => {
  const movie = { id: '1', title: 'Test Movie', genres: [] };
  render(<MovieThumbnail movie={movie} />);
  expect(screen.getByText('Test Movie')).toBeInTheDocument();
});

// Integration tests
test('search functionality', async () => {
  render(<App />);
  const searchInput = screen.getByPlaceholderText('Search movies...');
  fireEvent.change(searchInput, { target: { value: 'action' } });
  fireEvent.keyDown(searchInput, { key: 'Enter' });

  await waitFor(() => {
    expect(screen.getByText('Search Results')).toBeInTheDocument();
  });
});
```

---

*This architecture provides a scalable, maintainable foundation for modern full-stack ML applications. Each layer is designed to be modular, testable, and easy to extend.* 🌐