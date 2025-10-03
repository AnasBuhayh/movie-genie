# ML to Frontend Integration Guide

This guide provides step-by-step instructions for connecting ML model outputs to the backend API and then to the frontend interface. Each section includes precise code examples and implementation details.

## 🎯 Overview

The integration flow follows this pattern:
```
ML Models → Backend Services → API Endpoints → Frontend Data Service → UI Components
```

## 📋 Table of Contents

1. [ML Model Output Integration](#ml-model-output-integration)
2. [Backend Service Implementation](#backend-service-implementation)
3. [API Endpoint Creation](#api-endpoint-creation)
4. [Frontend Data Service Updates](#frontend-data-service-updates)
5. [Environment Configuration](#environment-configuration)
6. [Testing and Validation](#testing-and-validation)

---

## 1. ML Model Output Integration

### Example: BERT4Rec Personalized Recommendations

#### Step 1: ML Model Output Format
Your ML models should return data in this standardized format:

```python
# Expected output from BERT4Rec model
bert4rec_output = {
    'user_id': 123,
    'recommendations': [
        {
            'movie_id': 1001,
            'title': 'The Matrix',
            'score': 0.95,
            'rank': 1,
            'genres': ['Action', 'Sci-Fi'],
            'year': 1999,
            'poster_path': '/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg'
        },
        {
            'movie_id': 1002,
            'title': 'Inception',
            'score': 0.87,
            'rank': 2,
            'genres': ['Action', 'Thriller'],
            'year': 2010,
            'poster_path': '/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg'
        }
    ],
    'metadata': {
        'model_type': 'bert4rec',
        'sequence_length': 50,
        'total_candidates': 1000,
        'inference_time_ms': 45
    }
}
```

#### Step 2: Service Layer Integration
Update your recommendation service to use ML outputs:

```python
# movie_genie/backend/app/services/recommendation_service.py

class RecommendationService:
    def __init__(self):
        self.bert4rec_reranker = BERT4RecReranker()
        self.two_tower_reranker = TwoTowerReranker()

    def get_personalized_recommendations(self, user_id: str, limit: int = 10):
        """Get personalized recommendations using BERT4Rec model."""
        try:
            # 1. Get user interaction history
            user_history = self._get_user_history(user_id)

            # 2. Get ML model predictions
            ml_output = self.bert4rec_reranker.predict(
                user_id=int(user_id),
                sequence_length=50,
                num_recommendations=limit
            )

            # 3. Transform ML output to API format
            recommendations = self._transform_ml_to_api(ml_output, limit)

            return {
                'movies': recommendations,
                'recommendation_type': 'personalized',
                'user_context': {
                    'user_id': user_id,
                    'model_used': 'bert4rec',
                    'inference_time': ml_output.get('metadata', {}).get('inference_time_ms', 0)
                }
            }

        except Exception as e:
            logger.error(f"ML recommendation failed for user {user_id}: {e}")
            # Fallback to popular movies
            return self._get_popular_fallback(limit)

    def _transform_ml_to_api(self, ml_output: dict, limit: int) -> list:
        """Transform ML model output to standardized API format."""
        movies = []

        for item in ml_output.get('recommendations', [])[:limit]:
            movie = {
                'movieId': item['movie_id'],
                'title': item['title'],
                'genres': item.get('genres', []),
                'poster_path': item.get('poster_path'),
                'vote_average': self._score_to_rating(item['score']),
                'personalized_score': item['score'],
                'rank': item['rank'],
                'release_date': f"{item.get('year', 2020)}-01-01"
            }
            movies.append(movie)

        return movies

    def _score_to_rating(self, ml_score: float) -> float:
        """Convert ML confidence score (0-1) to rating (1-10)."""
        return round(1 + (ml_score * 9), 1)
```

---

## 2. Backend Service Implementation

### Example: Semantic Search Integration

#### Step 1: Search Service Update

```python
# movie_genie/backend/app/services/search_service.py

class SearchService:
    def __init__(self):
        self.semantic_engine = SemanticSearchEngine()

    def semantic_search(self, query: str, k: int = 20, user_context: dict = None):
        """Perform semantic search with ML-powered ranking."""
        try:
            # 1. Get semantic search results from ML model
            ml_results = self.semantic_engine.search(
                query=query,
                top_k=k,
                user_id=user_context.get('user_id') if user_context else None
            )

            # 2. Transform results for API response
            formatted_results = self._format_search_results(ml_results, query)

            return {
                'movies': formatted_results,
                'total': len(formatted_results),
                'query': query,
                'search_type': 'semantic',
                'ml_metadata': ml_results.get('metadata', {})
            }

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._fallback_search(query, k)

    def _format_search_results(self, ml_results: dict, query: str) -> list:
        """Format ML search results for API consumption."""
        movies = []

        for item in ml_results.get('results', []):
            movie = {
                'movieId': item['movie_id'],
                'title': item['title'],
                'overview': item.get('overview', ''),
                'genres': item.get('genres', []),
                'poster_path': item.get('poster_path'),
                'vote_average': item.get('rating', 0),
                'similarity_score': item['similarity_score'],
                'rank': item['rank']
            }
            movies.append(movie)

        return movies
```

---

## 3. API Endpoint Creation

### Example: Popular Movies with ML Ranking

#### Step 1: Create API Endpoint

```python
# movie_genie/backend/app/api/movies.py

from flask import Blueprint, request, jsonify
from ..services.movie_service import MovieService

movies_bp = Blueprint('movies', __name__)
movie_service = MovieService()

@movies_bp.route('/popular', methods=['GET'])
def get_popular_movies():
    """Get popular movies with optional ML-based personalization."""
    try:
        # Get parameters
        limit = int(request.args.get('limit', 20))
        user_id = request.args.get('user_id')  # Optional personalization

        # Get ML-powered popular movies
        result = movie_service.get_popular_movies(
            limit=limit,
            personalize_for_user=user_id
        )

        return jsonify({
            "success": True,
            "message": f"Retrieved {len(result['movies'])} popular movies",
            "data": result
        })

    except Exception as e:
        logger.error(f"Error getting popular movies: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to retrieve popular movies",
            "error": str(e)
        }), 500

@movies_bp.route('/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """Get detailed movie information with ML-enhanced metadata."""
    try:
        # Get base movie data
        movie = movie_service.get_movie_by_id(movie_id)

        if not movie:
            return jsonify({
                "success": False,
                "message": "Movie not found"
            }), 404

        # Enhance with ML-generated similar movies
        user_id = request.args.get('user_id')
        similar_movies = movie_service.get_similar_movies(
            movie_id=movie_id,
            user_context={'user_id': user_id} if user_id else None,
            limit=5
        )

        enhanced_movie = {
            **movie,
            'similar_movies': similar_movies,
            'ml_enhanced': True
        }

        return jsonify({
            "success": True,
            "data": enhanced_movie
        })

    except Exception as e:
        logger.error(f"Error getting movie {movie_id}: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to retrieve movie details"
        }), 500
```

---

## 4. Frontend Data Service Updates

### Step 1: Enable Real Data Sources

Update your environment configuration:

```bash
# movie_genie/frontend/.env.development

# Enable real data sources (set to true when ML integration is ready)
VITE_USE_REAL_POPULAR=true
VITE_USE_REAL_SEARCH=true
VITE_USE_REAL_RECOMMENDATIONS=true
VITE_USE_REAL_MOVIE_DETAILS=true
```

### Step 2: Update MovieDataService

```typescript
// movie_genie/frontend/src/services/movieDataService.ts

export class MovieDataService {

  // Enhanced popular movies with ML ranking
  static async getPopularMovies(limit: number = 20, userId?: string): Promise<MovieData[]> {
    if (DATA_SOURCE_CONFIG.popular) {
      try {
        console.log('🔄 Fetching ML-powered popular movies...');

        // Include user context for personalization
        const url = userId
          ? `${API_ENDPOINTS.POPULAR_MOVIES}?limit=${limit}&user_id=${userId}`
          : `${API_ENDPOINTS.POPULAR_MOVIES}?limit=${limit}`;

        const response = await fetch(url);
        const data = await response.json();

        if (data.success) {
          const movies = data.data.movies?.map(this.transformApiMovie) || [];
          console.log('✅ Got ML-powered popular movies:', movies.length);
          return movies;
        }

      } catch (error) {
        console.warn('⚠️ ML popular movies failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock popular movies');
    return this.getMockPopularMovies(limit);
  }

  // Enhanced search with ML semantic search
  static async searchMovies(query: string, limit: number = 20, userId?: string): Promise<SearchResults> {
    if (DATA_SOURCE_CONFIG.search && query.trim()) {
      try {
        console.log('🔄 Performing ML semantic search for:', query);

        const searchParams = new URLSearchParams({
          q: query,
          limit: limit.toString(),
          ...(userId && { user_id: userId })
        });

        const response = await fetch(`${API_ENDPOINTS.SEMANTIC_SEARCH}?${searchParams}`);
        const data = await response.json();

        if (data.success) {
          const movies = data.data.movies?.map(this.transformApiMovie) || [];
          console.log('✅ Got ML search results:', movies.length);

          return {
            movies: movies.slice(0, limit),
            total: data.data.total || movies.length,
            query,
            hasRealData: true,
            mlMetadata: data.data.ml_metadata || {}
          };
        }

      } catch (error) {
        console.warn('⚠️ ML search failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock search results for:', query);
    return this.getMockSearchResults(query, limit);
  }

  // Enhanced movie details with ML similar movies
  static async getMovieDetails(movieId: string, userId?: string): Promise<MovieData | null> {
    if (DATA_SOURCE_CONFIG.movieDetails) {
      try {
        console.log('🔄 Fetching ML-enhanced movie details for:', movieId);

        const url = userId
          ? `${API_ENDPOINTS.MOVIE_DETAILS(movieId)}?user_id=${userId}`
          : API_ENDPOINTS.MOVIE_DETAILS(movieId);

        const response = await fetch(url);
        const data = await response.json();

        if (data.success) {
          console.log('✅ Got ML-enhanced movie details');
          return this.transformApiMovie(data.data);
        }

      } catch (error) {
        console.warn('⚠️ ML movie details failed, falling back to mock data:', error);
      }
    }

    console.log('📝 Using mock movie details for:', movieId);
    return this.getMockMovieDetails(movieId);
  }

  // Transform API movie data (handles ML-specific fields)
  private static transformApiMovie(apiMovie: any): MovieData {
    return {
      id: apiMovie.movieId?.toString() || apiMovie.id?.toString(),
      title: apiMovie.title,
      poster_url: apiMovie.poster_path ? `https://image.tmdb.org/t/p/w500${apiMovie.poster_path}` : null,
      genres: apiMovie.genres || [],
      rating: apiMovie.vote_average,
      vote_average: apiMovie.vote_average,
      overview: apiMovie.overview,
      release_date: apiMovie.release_date,
      runtime: apiMovie.runtime,

      // ML-specific fields
      personalized_score: apiMovie.personalized_score,
      similarity_score: apiMovie.similarity_score,
      rank: apiMovie.rank,
      ml_enhanced: apiMovie.ml_enhanced || false,
      similar_movies: apiMovie.similar_movies?.map(this.transformApiMovie) || []
    };
  }
}
```

---

## 5. Environment Configuration

### Development vs Production Setup

#### Development Environment
```bash
# .env.development - Gradual ML integration
VITE_USE_REAL_POPULAR=true          # Start with popular movies
VITE_USE_REAL_SEARCH=false          # Keep search as mock for now
VITE_USE_REAL_RECOMMENDATIONS=false # Keep recommendations as mock
VITE_USE_REAL_MOVIE_DETAILS=false   # Keep details as mock
```

#### Production Environment
```bash
# .env.production - Full ML integration
VITE_USE_REAL_POPULAR=true
VITE_USE_REAL_SEARCH=true
VITE_USE_REAL_RECOMMENDATIONS=true
VITE_USE_REAL_MOVIE_DETAILS=true
```

---

## 6. Testing and Validation

### Step 1: Backend Testing

```python
# test_ml_integration.py

import pytest
from movie_genie.backend.app.services.recommendation_service import RecommendationService

def test_ml_personalized_recommendations():
    """Test ML-powered personalized recommendations."""
    service = RecommendationService()

    # Test with real user ID
    result = service.get_personalized_recommendations(user_id="123", limit=10)

    assert result['recommendation_type'] == 'personalized'
    assert len(result['movies']) <= 10
    assert all('personalized_score' in movie for movie in result['movies'])
    assert result['user_context']['model_used'] == 'bert4rec'

def test_ml_search_integration():
    """Test ML-powered semantic search."""
    service = SearchService()

    result = service.semantic_search(query="action movies", k=20)

    assert result['search_type'] == 'semantic'
    assert len(result['movies']) <= 20
    assert all('similarity_score' in movie for movie in result['movies'])
```

### Step 2: Frontend Testing

```javascript
// Test real vs mock data switching
console.log('Data source status:', MovieDataService.getDataSourceStatus());

// Test ML-powered popular movies
const popularMovies = await MovieDataService.getPopularMovies(10, '123');
console.log('Popular movies with ML:', popularMovies);

// Test ML-powered search
const searchResults = await MovieDataService.searchMovies('action', 20, '123');
console.log('Search results with ML:', searchResults);
```

### Step 3: UI Validation

1. **Check Data Source Indicators**: Look for "🌐 Real Data" badges in search results
2. **Verify ML Scores**: Check hover tooltips show personalized/similarity scores
3. **Test Fallback Behavior**: Disable ML services and verify graceful degradation
4. **Performance Monitoring**: Check ML inference times in network tab

---

## 🚀 Quick Start Checklist

### Phase 1: Popular Movies Integration
- [ ] Update `MovieService.get_popular_movies()` to use ML ranking
- [ ] Set `VITE_USE_REAL_POPULAR=true`
- [ ] Test homepage carousels show real data
- [ ] Verify fallback works when ML service fails

### Phase 2: Search Integration
- [ ] Update `SearchService.semantic_search()` to use ML embeddings
- [ ] Set `VITE_USE_REAL_SEARCH=true`
- [ ] Test search grid shows semantic results
- [ ] Verify data source indicator shows "🌐 Real Data"

### Phase 3: Recommendations Integration
- [ ] Update `RecommendationService.get_personalized_recommendations()`
- [ ] Set `VITE_USE_REAL_RECOMMENDATIONS=true`
- [ ] Test personalized carousels show ML results
- [ ] Verify user context is passed correctly

### Phase 4: Movie Details Integration
- [ ] Update `MovieService.get_movie_by_id()` to include similar movies
- [ ] Set `VITE_USE_REAL_MOVIE_DETAILS=true`
- [ ] Test movie details panel shows enhanced data
- [ ] Verify similar movies section appears

---

## 🔧 Debugging Tips

### Common Issues and Solutions

1. **ML Model Not Loading**
   ```python
   # Add to service initialization
   if not self.bert4rec_reranker:
       logger.error("BERT4Rec model failed to load - check model files")
       # Initialize fallback service
   ```

2. **API Response Format Mismatch**
   ```javascript
   // Add response validation
   if (!data.success || !data.data) {
       console.warn('Invalid API response format:', data);
       throw new Error('Invalid response format');
   }
   ```

3. **Performance Issues**
   ```python
   # Add caching for expensive ML operations
   @lru_cache(maxsize=1000)
   def get_cached_recommendations(user_id: str, cache_key: str):
       return self.bert4rec_reranker.predict(user_id)
   ```

4. **Frontend Data Source Confusion**
   ```typescript
   // Add clear logging
   console.log(`Using ${DATA_SOURCE_CONFIG.popular ? 'REAL' : 'MOCK'} data for popular movies`);
   ```

This guide provides the complete integration path from your ML models to the user interface. Follow each phase sequentially, testing thoroughly before moving to the next integration point.