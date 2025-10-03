# 📚 API Reference

Complete documentation for Movie Genie's REST API endpoints, including request/response formats, authentication, and examples.

## 🌐 API Overview

**Base URL**: `http://127.0.0.1:5001/api`

**Content Type**: `application/json`

**Response Format**: All endpoints return JSON with the following structure:
```json
{
  "success": boolean,
  "message": string,
  "data": object | array | null,
  "timestamp": string (ISO 8601)
}
```

---

## 🔧 System Endpoints

### Health Check

Check if the API server is running and healthy.

#### `GET /health`

**Description**: Returns the current system status and health information.

**Parameters**: None

**Response**:
```json
{
  "success": true,
  "message": "System is healthy",
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600,
    "models_loaded": {
      "bert4rec": true,
      "two_tower": true,
      "semantic_search": true
    },
    "database_status": "connected"
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/health
```

---

## 👥 User Endpoints

### Get User Information

Retrieve information about a specific user or current user context.

#### `GET /users/info`

**Description**: Returns user information and statistics.

**Parameters**:
- `user_id` (query, optional): Specific user ID to query (1-610)

**Response**:
```json
{
  "success": true,
  "message": "User information retrieved successfully",
  "data": {
    "current_user": 123,
    "total_users": 943,
    "user_range": {
      "min": 1,
      "max": 610
    },
    "user_stats": {
      "ratings_count": 150,
      "avg_rating": 3.8,
      "favorite_genres": ["Action", "Sci-Fi"],
      "last_activity": "2024-01-01T10:30:00.000Z"
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
# Get general user info
curl http://127.0.0.1:5001/api/users/info

# Get specific user info
curl "http://127.0.0.1:5001/api/users/info?user_id=123"
```

---

## 🎬 Movie Endpoints

### Get Popular Movies

Retrieve a list of popular movies with optional personalization.

#### `GET /movies/popular`

**Description**: Returns popular movies, optionally personalized for a specific user.

**Parameters**:
- `limit` (query, optional): Number of movies to return (default: 20, max: 100)
- `user_id` (query, optional): User ID for personalization (1-610)

**Response**:
```json
{
  "success": true,
  "message": "Retrieved 20 popular movies",
  "data": {
    "movies": [
      {
        "movieId": 1,
        "title": "Toy Story",
        "genres": ["Animation", "Children's", "Comedy"],
        "poster_path": "/w4pJJ6VsZPNHdKJxCZZLfYRyPac.jpg",
        "vote_average": 8.3,
        "release_date": "1995-10-30",
        "overview": "A cowboy doll is profoundly threatened...",
        "runtime": 81,
        "personalized_score": 0.92,
        "rank": 1
      }
    ],
    "total": 20,
    "recommendation_type": "popular",
    "personalized": true,
    "user_context": {
      "user_id": "123",
      "model_used": "two_tower",
      "inference_time": 15
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
# Get popular movies
curl "http://127.0.0.1:5001/api/movies/popular?limit=10"

# Get personalized popular movies
curl "http://127.0.0.1:5001/api/movies/popular?limit=10&user_id=123"
```

### Get Movie Details

Retrieve detailed information about a specific movie.

#### `GET /movies/{movie_id}`

**Description**: Returns comprehensive movie information including similar movies.

**Parameters**:
- `movie_id` (path): Movie ID (integer)
- `user_id` (query, optional): User ID for personalized similar movies

**Response**:
```json
{
  "success": true,
  "message": "Movie details retrieved successfully",
  "data": {
    "movieId": 1,
    "title": "Toy Story",
    "genres": ["Animation", "Children's", "Comedy"],
    "poster_path": "/w4pJJ6VsZPNHdKJxCZZLfYRyPac.jpg",
    "vote_average": 8.3,
    "release_date": "1995-10-30",
    "overview": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
    "runtime": 81,
    "director": "John Lasseter",
    "cast": ["Tom Hanks", "Tim Allen", "Don Rickles"],
    "similar_movies": [
      {
        "movieId": 2,
        "title": "Jumanji",
        "similarity_score": 0.85,
        "genres": ["Adventure", "Children's", "Fantasy"]
      }
    ],
    "ml_enhanced": true
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
# Get movie details
curl http://127.0.0.1:5001/api/movies/1

# Get movie details with personalized similar movies
curl "http://127.0.0.1:5001/api/movies/1?user_id=123"
```

---

## 🔍 Search Endpoints

### Semantic Search

Search for movies using natural language queries.

#### `GET /search/semantic`

**Description**: Performs semantic search using ML-powered text embeddings.

**Parameters**:
- `q` (query, required): Search query string
- `limit` (query, optional): Number of results to return (default: 20, max: 50)
- `user_id` (query, optional): User ID for personalized ranking

**Response**:
```json
{
  "success": true,
  "message": "Semantic search completed successfully",
  "data": {
    "movies": [
      {
        "movieId": 1,
        "title": "Toy Story",
        "overview": "A cowboy doll is profoundly threatened...",
        "genres": ["Animation", "Children's", "Comedy"],
        "poster_path": "/w4pJJ6VsZPNHdKJxCZZLfYRyPac.jpg",
        "vote_average": 8.3,
        "similarity_score": 0.95,
        "rank": 1
      }
    ],
    "total": 15,
    "query": "animated movies for kids",
    "search_type": "semantic",
    "ml_metadata": {
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
      "search_time_ms": 25,
      "total_candidates": 1682
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
# Basic semantic search
curl "http://127.0.0.1:5001/api/search/semantic?q=action%20movies%20with%20robots"

# Personalized semantic search
curl "http://127.0.0.1:5001/api/search/semantic?q=funny%20movies&limit=10&user_id=123"
```

### Traditional Search

Search for movies by title, genre, or keywords.

#### `GET /search/traditional`

**Description**: Performs traditional text-based search using database queries.

**Parameters**:
- `q` (query, required): Search query string
- `limit` (query, optional): Number of results to return (default: 20, max: 50)
- `genre` (query, optional): Filter by specific genre
- `year` (query, optional): Filter by release year

**Response**:
```json
{
  "success": true,
  "message": "Traditional search completed successfully",
  "data": {
    "movies": [
      {
        "movieId": 1,
        "title": "Toy Story",
        "genres": ["Animation", "Children's", "Comedy"],
        "poster_path": "/w4pJJ6VsZPNHdKJxCZZLfYRyPac.jpg",
        "vote_average": 8.3,
        "release_date": "1995-10-30",
        "match_score": 0.98,
        "match_type": "title"
      }
    ],
    "total": 5,
    "query": "toy story",
    "search_type": "traditional",
    "filters": {
      "genre": null,
      "year": null
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
# Basic traditional search
curl "http://127.0.0.1:5001/api/search/traditional?q=star%20wars"

# Filtered search
curl "http://127.0.0.1:5001/api/search/traditional?q=action&genre=Action&year=1995"
```

---

## 🤖 Recommendation Endpoints

### Personalized Recommendations

Get personalized movie recommendations for a specific user.

#### `GET /recommendations/personalized`

**Description**: Returns ML-powered personalized recommendations using BERT4Rec model.

**Parameters**:
- `user_id` (query, required): User ID for recommendations (1-610)
- `limit` (query, optional): Number of recommendations (default: 10, max: 50)
- `model` (query, optional): ML model to use ('bert4rec', 'two_tower', 'hybrid')

**Response**:
```json
{
  "success": true,
  "message": "Personalized recommendations generated successfully",
  "data": {
    "movies": [
      {
        "movieId": 1,
        "title": "Toy Story",
        "genres": ["Animation", "Children's", "Comedy"],
        "poster_path": "/w4pJJ6VsZPNHdKJxCZZLfYRyPac.jpg",
        "vote_average": 8.3,
        "personalized_score": 0.95,
        "rank": 1,
        "prediction_confidence": 0.87
      }
    ],
    "total": 10,
    "recommendation_type": "personalized",
    "user_context": {
      "user_id": "123",
      "model_used": "bert4rec",
      "sequence_length": 50,
      "user_history_size": 45,
      "inference_time_ms": 75
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
# Get personalized recommendations
curl "http://127.0.0.1:5001/api/recommendations/personalized?user_id=123"

# Get recommendations with specific model
curl "http://127.0.0.1:5001/api/recommendations/personalized?user_id=123&limit=20&model=bert4rec"
```

### Similar Movies

Get movies similar to a specific movie.

#### `GET /recommendations/similar/{movie_id}`

**Description**: Returns movies similar to the specified movie using content-based filtering.

**Parameters**:
- `movie_id` (path): Reference movie ID
- `limit` (query, optional): Number of similar movies (default: 10, max: 20)
- `user_id` (query, optional): User ID for personalized ranking

**Response**:
```json
{
  "success": true,
  "message": "Similar movies found successfully",
  "data": {
    "reference_movie": {
      "movieId": 1,
      "title": "Toy Story"
    },
    "similar_movies": [
      {
        "movieId": 2,
        "title": "Jumanji",
        "genres": ["Adventure", "Children's", "Fantasy"],
        "similarity_score": 0.85,
        "similarity_reason": "genre_and_content",
        "rank": 1
      }
    ],
    "total": 10,
    "recommendation_type": "similar",
    "similarity_method": "content_based"
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
# Get similar movies
curl http://127.0.0.1:5001/api/recommendations/similar/1

# Get personalized similar movies
curl "http://127.0.0.1:5001/api/recommendations/similar/1?user_id=123&limit=5"
```

---

## 📊 Analytics Endpoints

### User Analytics

Get analytics and statistics for user behavior.

#### `GET /analytics/user/{user_id}`

**Description**: Returns detailed analytics for a specific user's movie preferences and behavior.

**Parameters**:
- `user_id` (path): User ID for analytics

**Response**:
```json
{
  "success": true,
  "message": "User analytics retrieved successfully",
  "data": {
    "user_id": 123,
    "rating_statistics": {
      "total_ratings": 150,
      "avg_rating": 3.8,
      "rating_distribution": {
        "1": 5,
        "2": 15,
        "3": 45,
        "4": 60,
        "5": 25
      }
    },
    "genre_preferences": [
      {
        "genre": "Action",
        "count": 35,
        "avg_rating": 4.1
      },
      {
        "genre": "Sci-Fi",
        "count": 28,
        "avg_rating": 4.3
      }
    ],
    "activity_timeline": {
      "first_rating": "1997-01-01T00:00:00.000Z",
      "last_rating": "1998-12-31T23:59:59.000Z",
      "most_active_month": "1998-06"
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/analytics/user/123
```

---

## ⚠️ Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "success": false,
  "message": "Human-readable error description",
  "error": {
    "code": "ERROR_CODE",
    "details": "Technical error details",
    "field": "field_name (for validation errors)"
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### HTTP Status Codes

| Status Code | Description | Example |
|-------------|-------------|---------|
| `200` | Success | Successful API call |
| `400` | Bad Request | Invalid parameters or malformed request |
| `404` | Not Found | Movie or user not found |
| `422` | Validation Error | Invalid user ID or parameter values |
| `500` | Internal Server Error | Database connection failed or ML model error |
| `503` | Service Unavailable | ML models not loaded or system maintenance |

### Common Error Codes

#### Validation Errors (400)
```json
{
  "success": false,
  "message": "Invalid user ID",
  "error": {
    "code": "INVALID_USER_ID",
    "details": "User ID must be between 1 and 610",
    "field": "user_id"
  }
}
```

#### Not Found Errors (404)
```json
{
  "success": false,
  "message": "Movie not found",
  "error": {
    "code": "MOVIE_NOT_FOUND",
    "details": "No movie found with ID 99999"
  }
}
```

#### ML Model Errors (500)
```json
{
  "success": false,
  "message": "Recommendation model unavailable",
  "error": {
    "code": "ML_MODEL_ERROR",
    "details": "BERT4Rec model failed to load"
  }
}
```

---

## 🔧 Rate Limiting

### Rate Limits

| Endpoint Category | Rate Limit | Window |
|------------------|------------|--------|
| **Health/Info** | 60 requests | 1 minute |
| **Search** | 30 requests | 1 minute |
| **Recommendations** | 20 requests | 1 minute |
| **Movie Details** | 100 requests | 1 minute |

### Rate Limit Headers

```http
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1640995200
```

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "message": "Rate limit exceeded",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "details": "Maximum 30 requests per minute for search endpoints"
  }
}
```

---

## 🧪 Testing the API

### Using cURL

```bash
# Test all endpoints
curl http://127.0.0.1:5001/api/health
curl http://127.0.0.1:5001/api/users/info
curl "http://127.0.0.1:5001/api/movies/popular?limit=5"
curl "http://127.0.0.1:5001/api/search/semantic?q=action%20movies"
curl "http://127.0.0.1:5001/api/recommendations/personalized?user_id=123"
```

### Using Python

```python
import requests

# API base URL
API_URL = "http://127.0.0.1:5001/api"

# Test search
response = requests.get(f"{API_URL}/search/semantic", params={
    "q": "funny movies for kids",
    "limit": 10,
    "user_id": 123
})

data = response.json()
if data["success"]:
    movies = data["data"]["movies"]
    print(f"Found {len(movies)} movies")
else:
    print(f"Error: {data['message']}")
```

### Using JavaScript

```javascript
// API client example
class MovieAPI {
  constructor(baseURL = 'http://127.0.0.1:5001/api') {
    this.baseURL = baseURL;
  }

  async searchMovies(query, limit = 20, userId = null) {
    const params = new URLSearchParams({ q: query, limit });
    if (userId) params.append('user_id', userId);

    const response = await fetch(`${this.baseURL}/search/semantic?${params}`);
    const data = await response.json();

    if (!data.success) {
      throw new Error(data.message);
    }

    return data.data;
  }

  async getRecommendations(userId, limit = 10) {
    const response = await fetch(
      `${this.baseURL}/recommendations/personalized?user_id=${userId}&limit=${limit}`
    );
    const data = await response.json();

    if (!data.success) {
      throw new Error(data.message);
    }

    return data.data;
  }
}

// Usage
const api = new MovieAPI();
const movies = await api.searchMovies("action movies", 10, 123);
```

---

*This API reference provides complete documentation for integrating with Movie Genie's backend services. All endpoints are designed to be RESTful, well-documented, and easy to test.* 📚