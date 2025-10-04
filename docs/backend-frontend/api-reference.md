# 📚 API Reference

Complete documentation for Movie Genie's REST API endpoints, including request/response formats, data schemas, and examples.

## 🌐 API Overview

**Base URL**: `http://127.0.0.1:5001/api`

**Content Type**: `application/json`

**Response Format**: All endpoints return JSON with a consistent wrapper structure:
```json
{
  "success": boolean,
  "message": string,
  "data": object | array | null,
  "error_code": string (optional, only on errors),
  "details": any (optional, only on errors)
}
```

### Important Notes

- **Response Unwrapping**: The frontend automatically unwraps the `data` field from the response wrapper
- **Poster URLs**: All `poster_path` values are relative (e.g., `/xyz.jpg`) and need to be prepended with `https://image.tmdb.org/t/p/w500` for display
- **Genre Format**: Genres are always returned as string arrays, never pipe-separated strings
- **User IDs**: Valid user IDs range from 1-610 (MovieLens small dataset)

---

## 📊 Data Schemas

### Movie Object Schema

All movie objects returned by the API follow this schema:

```typescript
{
  movieId: number,                    // Primary identifier
  title: string,                      // Movie title
  overview: string | null,            // Plot summary
  genres: string[],                   // Array of genre names (e.g., ["Action", "Sci-Fi"])
  vote_average: number | null,        // TMDB rating (0-10 scale)
  vote_count: number | null,          // Number of votes
  runtime: number | null,             // Duration in minutes
  release_date: string | null,        // ISO date format (YYYY-MM-DD)
  poster_path: string | null,         // Relative path to poster image

  // Search-specific fields (only in search results)
  similarity_score?: number,          // Semantic similarity (0-1 scale, higher = better match)
  rank?: number,                      // Position in ranked results

  // Recommendation-specific fields
  personalized_score?: number,        // Personalization score from ML model
  prediction_confidence?: number,     // Model confidence (0-1 scale)

  // User interaction fields (only when applicable)
  watched?: boolean,                  // User has watched this movie
  liked?: boolean,                    // User liked this movie
  disliked?: boolean                  // User disliked this movie
}
```

### Search Response Schema

```typescript
{
  movies: Movie[],                    // Array of movie objects
  total: number,                      // Total number of results
  query: string,                      // Original search query
  search_type: "semantic" | "traditional"
}
```

### Recommendation Response Schema

```typescript
{
  movies: Movie[],                    // Array of recommended movies
  total: number,                      // Total number of recommendations
  recommendation_type: string         // Type of recommendation
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
    "version": "1.0.0"
  }
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/health
```

---

## 👥 User Endpoints

### Get User Information

Retrieve information about available users in the system.

#### `GET /users/info`

**Description**: Returns user range and statistics for the MovieLens dataset.

**Parameters**: None

**Response**:
```json
{
  "success": true,
  "message": "User information retrieved",
  "data": {
    "total_users": 610,
    "user_range": {
      "min": 1,
      "max": 610
    },
    "description": "MovieLens small dataset users"
  }
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/users/info
```

---

### Get User Profile

Retrieve detailed profile for a specific user including interaction history.

#### `GET /users/{user_id}/profile`

**Description**: Returns user profile with complete interaction history for ML model personalization.

**Parameters**:
- `user_id` (path, required): User ID (1-610)

**Response**:
```json
{
  "success": true,
  "message": "User profile retrieved",
  "data": {
    "user_id": 123,
    "interaction_history": [
      {
        "movieId": 1,
        "rating": 5.0,
        "timestamp": "1997-01-01T00:00:00Z"
      },
      {
        "movieId": 2,
        "rating": 3.0,
        "timestamp": "1997-01-02T00:00:00Z"
      }
    ],
    "total_interactions": 150
  }
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/users/123/profile
```

**Usage**: The interaction history is used by BERT4Rec for sequential recommendations and personalized search ranking.

---

### Get User's Watched Movies

Retrieve movies that the user has watched (based on rating history).

#### `GET /users/{user_id}/watched`

**Description**: Returns the most recent movies the user has interacted with, sorted by timestamp.

**Parameters**:
- `user_id` (path, required): User ID (1-610)
- `limit` (query, optional): Number of movies to return (default: 20, max: 100)

**Response**:
```json
{
  "success": true,
  "message": "Retrieved 20 watched movies for user 123",
  "data": {
    "movies": [
      {
        "movieId": 1,
        "title": "Toy Story",
        "overview": "A cowboy doll is profoundly threatened...",
        "genres": ["Animation", "Adventure", "Comedy"],
        "poster_path": "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
        "vote_average": 7.7,
        "vote_count": 5415,
        "runtime": 81,
        "release_date": "1995-10-30",
        "watched": true
      }
    ],
    "total": 20
  }
}
```

**Example**:
```bash
# Get last 20 watched movies
curl http://127.0.0.1:5001/api/users/123/watched

# Get last 8 watched movies
curl "http://127.0.0.1:5001/api/users/123/watched?limit=8"
```

**Implementation Details**:
- Loads user interaction data from `sequences_with_metadata.parquet`
- Gets unique movie IDs from most recent interactions
- Enriches with full movie metadata from `content_features.parquet`
- Marks all movies with `watched: true` flag

---

### Get User's Historical Interest

Get personalized recommendations based on user's favorite genres.

#### `GET /users/{user_id}/historical-interest`

**Description**: Analyzes user's rating history to identify top 3 favorite genres, then returns unwatched movies from those genres.

**Parameters**:
- `user_id` (path, required): User ID (1-610)
- `limit` (query, optional): Number of movies to return (default: 20, max: 100)

**Response**:
```json
{
  "success": true,
  "message": "Retrieved 20 historical interest movies for user 123",
  "data": {
    "movies": [
      {
        "movieId": 2,
        "title": "Jumanji",
        "overview": "When siblings Judy and Peter discover...",
        "genres": ["Adventure", "Fantasy", "Family"],
        "poster_path": "/vgpXmVaVyUL7GGiDeiK1mKEKzcX.jpg",
        "vote_average": 6.9,
        "vote_count": 2413,
        "runtime": 104,
        "release_date": "1995-12-15"
      }
    ],
    "total": 20,
    "favorite_genres": ["Adventure", "Fantasy", "Action"]
  }
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/users/123/historical-interest
```

**Algorithm**:
1. Load all user interactions with genre information
2. Count genre occurrences across rated movies
3. Identify top 3 most-watched genres
4. Find unwatched movies matching those genres (via set difference)
5. Sort by popularity (vote_count) and return top N

**Genre Parsing**: Handles both pipe-separated strings (`"Action|Adventure"`) and array formats.

---

## 🎬 Movie Endpoints

### Get Popular Movies

Retrieve globally popular movies sorted by vote count.

#### `GET /movies/popular`

**Description**: Returns most popular movies from the dataset based on TMDB vote counts.

**Parameters**:
- `limit` (query, optional): Number of movies to return (default: 20, max: 100)

**Response**:
```json
{
  "success": true,
  "message": "Retrieved 20 popular movies",
  "data": {
    "movies": [
      {
        "movieId": 296,
        "title": "Pulp Fiction",
        "overview": "A burger-loving hit man...",
        "genres": ["Thriller", "Crime"],
        "poster_path": "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
        "vote_average": 8.5,
        "vote_count": 8670,
        "runtime": 154,
        "release_date": "1994-09-10"
      }
    ],
    "total": 20
  }
}
```

**Example**:
```bash
curl "http://127.0.0.1:5001/api/movies/popular?limit=10"
```

**Sorting**: Movies are sorted by `vote_count` descending (most voted = most popular).

---

### Get Movie Details

Retrieve detailed information about a specific movie.

#### `GET /movies/{movie_id}`

**Description**: Returns comprehensive movie information from content features dataset.

**Parameters**:
- `movie_id` (path, required): Movie ID (integer)

**Response**:
```json
{
  "success": true,
  "message": "Movie details retrieved",
  "data": {
    "movieId": 1,
    "title": "Toy Story",
    "overview": "Led by Woody, Andy's toys live happily...",
    "genres": ["Animation", "Adventure", "Comedy"],
    "poster_path": "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
    "vote_average": 7.7,
    "vote_count": 5415,
    "runtime": 81,
    "release_date": "1995-10-30",
    "budget": 30000000,
    "revenue": 373554033,
    "tagline": "The adventure takes off!",
    "original_language": "en"
  }
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/movies/1
```

---

## 🔍 Search Endpoints

### Semantic Search

Search for movies using natural language queries with ML-powered semantic understanding.

#### `GET /search/semantic`

**Description**: Performs semantic search using sentence-transformers embeddings to find movies matching the query's semantic meaning, not just keywords.

**Parameters**:
- `q` (query, required): Natural language search query
- `k` (query, optional): Number of results to return (default: 20, max: 100)
- `user_id` (query, optional): User ID for personalized ranking with BERT4Rec reranking

**Response**:
```json
{
  "success": true,
  "message": "Found 15 movies for 'action movies with robots'",
  "data": {
    "movies": [
      {
        "movieId": 589,
        "title": "Terminator 2: Judgment Day",
        "overview": "Nearly 10 years have passed since Sarah Connor...",
        "genres": ["Action", "Thriller", "Science Fiction"],
        "poster_path": "/5M0j0B18abtBI5gi2RhfjjurTqb.jpg",
        "vote_average": 8.1,
        "vote_count": 3513,
        "runtime": 137,
        "release_date": "1991-07-03",
        "similarity_score": 0.7234,
        "rank": 1
      }
    ],
    "total": 15,
    "query": "action movies with robots",
    "search_type": "semantic",
    "personalized": false
  }
}
```

**Example**:
```bash
# Basic semantic search
curl "http://127.0.0.1:5001/api/search/semantic?q=funny%20animated%20movies"

# Personalized semantic search
curl "http://127.0.0.1:5001/api/search/semantic?q=dark%20thriller&k=10&user_id=123"

# Multi-word queries work great
curl "http://127.0.0.1:5001/api/search/semantic?q=movies%20about%20space%20exploration"
```

**How It Works**:

1. **Query Encoding**: User query is encoded into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`
2. **Similarity Search**: Compute cosine similarity between query vector and all 9,742 pre-computed movie embeddings
3. **Candidate Retrieval**: Get top 3×k candidates (e.g., 60 candidates for k=20)
4. **Reranking** (if user_id provided):
   - Use BERT4Rec model to rerank based on user's interaction history
   - Combine semantic similarity (60%) + personalization score (40%)
5. **Return Top-k**: Return final top k results

**Similarity Scores**:
- Range: 0.0 to 1.0
- Higher = better match
- Typical good matches: 0.5-0.8
- Perfect matches rare (would be ~1.0)

**Model Details**:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Configuration**: `configs/semantic_search.yaml`
- **Embeddings**: Pre-computed in `data/processed/content_features.parquet`

**When Regeneration Needed**:
- After changing embedding model in config
- After updating movie dataset
- Run: `dvc repro content_features`

---

### Traditional Search

Search for movies by title using case-insensitive partial matching.

#### `GET /search/traditional` or `GET /search/`

**Description**: Performs traditional title-based search when semantic search fails or as a fallback. Uses case-insensitive substring matching.

**Parameters**:
- `q` (query, required): Search query string
- `k` (query, optional): Number of results to return (default: 20, max: 100)

**Response**:
```json
{
  "success": true,
  "message": "Found 5 movies for 'batman'",
  "data": {
    "movies": [
      {
        "movieId": 268,
        "title": "Batman",
        "overview": "The Dark Knight of Gotham City...",
        "genres": ["Fantasy", "Action"],
        "poster_path": "/kBf3g9crrADGMc2AMAMlLBgSm2h.jpg",
        "vote_average": 7.2,
        "vote_count": 1511,
        "runtime": 126,
        "release_date": "1989-06-23",
        "rank": 1
      },
      {
        "movieId": 364,
        "title": "Batman Returns",
        "overview": "Having defeated the Joker...",
        "genres": ["Action", "Fantasy"],
        "poster_path": "/jKBjeXM7iBBV9UkUcOXx3m7FSHY.jpg",
        "vote_average": 6.8,
        "vote_count": 1084,
        "runtime": 126,
        "release_date": "1992-06-19",
        "rank": 2
      }
    ],
    "total": 5,
    "query": "batman",
    "search_type": "traditional"
  }
}
```

**Example**:
```bash
# Search by title
curl "http://127.0.0.1:5001/api/search/traditional?q=star%20wars"

# Search with limit
curl "http://127.0.0.1:5001/api/search/traditional?q=terminator&k=5"
```

**Search Algorithm**:
- Case-insensitive: "BATMAN" matches "Batman"
- Substring match: "term" matches "Terminator 2"
- Uses MovieService.search_movies_by_title()
- No semantic understanding (exact keyword matching only)

**When Used**:
- Automatically as fallback when semantic search fails
- Frontend tries semantic first, then traditional
- Best for exact title searches

---

### Search Status

Check semantic search engine health and configuration.

#### `GET /search/status`

**Description**: Returns status information about the search engine, including model availability and statistics.

**Parameters**: None

**Response**:
```json
{
  "success": true,
  "message": "Search service status retrieved",
  "data": {
    "semantic_engine_available": true,
    "config_path": "/path/to/configs/semantic_search.yaml",
    "service": "SearchService",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "total_movies_indexed": 9742,
    "embedding_dimension": 384
  }
}
```

**Example**:
```bash
curl http://127.0.0.1:5001/api/search/status
```

**Troubleshooting**:
- If `semantic_engine_available: false`, check logs for initialization errors
- Common issues: Missing torch/transformers, embedding dimension mismatch
- See troubleshooting docs for solutions

---

## 🤖 Recommendation Endpoints

### Personalized Recommendations

Get personalized movie recommendations using BERT4Rec sequential model.

#### `POST /recommendations/personalized`

**Description**: Returns personalized recommendations based on user's interaction history using BERT4Rec model.

**Parameters**:
- Body (JSON):
```json
{
  "user_id": "123",
  "interaction_history": [
    {"movieId": 1, "rating": 5.0, "timestamp": "1997-01-01T00:00:00Z"},
    {"movieId": 2, "rating": 4.0, "timestamp": "1997-01-02T00:00:00Z"}
  ]
}
```

**Response**:
```json
{
  "success": true,
  "message": "Recommendations generated",
  "data": {
    "movies": [
      {
        "movieId": 3,
        "title": "Grumpier Old Men",
        "overview": "A family wedding reignites...",
        "genres": ["Romance", "Comedy"],
        "poster_path": "/1FSXpj5e8l4KH6nVFO5SPUeraOt.jpg",
        "vote_average": 6.5,
        "vote_count": 92,
        "runtime": 101,
        "release_date": "1995-12-22",
        "personalized_score": 0.89,
        "rank": 1
      }
    ],
    "total": 10
  }
}
```

**Example**:
```bash
curl -X POST http://127.0.0.1:5001/api/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "123",
    "interaction_history": [
      {"movieId": 1, "rating": 5.0, "timestamp": "1997-01-01T00:00:00Z"}
    ]
  }'
```

**Model**: BERT4Rec (Bidirectional Encoder Representations from Transformers for Recommender)
**Input**: User's sequential interaction history
**Output**: Top-k personalized recommendations

---

## 💬 Feedback Endpoints

### Submit Movie Rating

Submit a user's rating for a movie.

#### `POST /feedback/rating`

**Description**: Record user feedback with a numerical rating.

**Parameters**:
- Body (JSON):
```json
{
  "user_id": "123",
  "movie_id": 1,
  "rating": 4.5
}
```

**Response**:
```json
{
  "success": true,
  "message": "Rating submitted successfully"
}
```

**Example**:
```bash
curl -X POST http://127.0.0.1:5001/api/feedback/rating \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123", "movie_id": 1, "rating": 4.5}'
```

---

## ⚠️ Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "success": false,
  "message": "Human-readable error description",
  "error_code": "ERROR_CODE",
  "details": "Technical error details (optional)"
}
```

### Common Error Codes

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `MOVIE_NOT_FOUND` | 404 | Movie ID doesn't exist |
| `USER_NOT_FOUND` | 404 | User ID not in valid range (1-610) |
| `SEARCH_ENGINE_ERROR` | 503 | Semantic search engine unavailable |
| `ML_MODEL_ERROR` | 503 | BERT4Rec or Two-Tower model failed |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

### Example Error Responses

**Validation Error**:
```json
{
  "success": false,
  "message": "Query parameter 'q' is required",
  "error_code": "VALIDATION_ERROR",
  "details": "Search query cannot be empty"
}
```

**Service Unavailable**:
```json
{
  "success": false,
  "message": "Search failed: Search engine not available",
  "error_code": "SEARCH_ENGINE_ERROR",
  "details": {
    "movies": [],
    "total": 0,
    "error": "Search engine not available"
  }
}
```

---

## 🧪 Testing the API

### Quick Test Commands

```bash
# Health check
curl http://127.0.0.1:5001/api/health

# User info
curl http://127.0.0.1:5001/api/users/info

# Get user profile
curl http://127.0.0.1:5001/api/users/1/profile

# Popular movies
curl "http://127.0.0.1:5001/api/movies/popular?limit=5"

# Semantic search
curl "http://127.0.0.1:5001/api/search/semantic?q=action%20movies&k=5"

# Traditional search
curl "http://127.0.0.1:5001/api/search/traditional?q=batman"

# Movie details
curl http://127.0.0.1:5001/api/movies/1

# User's watched movies
curl http://127.0.0.1:5001/api/users/1/watched

# Historical interest
curl http://127.0.0.1:5001/api/users/1/historical-interest

# Search status
curl http://127.0.0.1:5001/api/search/status
```

### Testing with Python

```python
import requests

API_BASE = "http://127.0.0.1:5001/api"

# Test semantic search
response = requests.get(f"{API_BASE}/search/semantic", params={
    "q": "funny animated movies for kids",
    "k": 10,
    "user_id": "1"
})

data = response.json()
if data["success"]:
    movies = data["data"]["movies"]
    for movie in movies:
        print(f"{movie['title']} - Score: {movie.get('similarity_score', 'N/A')}")
else:
    print(f"Error: {data['message']}")
```

### Frontend Integration Pattern

```typescript
// API client in frontend/src/lib/api.ts
class MovieGenieAPI {
  static async searchMovies(query: string, useSemanticSearch: boolean, userId?: string) {
    const endpoint = useSemanticSearch ? '/search/semantic' : '/search/traditional';
    const url = `${API_BASE}${endpoint}?q=${encodeURIComponent(query)}`;

    const response = await fetch(url);
    const jsonResponse = await response.json();

    // Unwrap the {success, data, message} wrapper
    if (jsonResponse.success && jsonResponse.data !== undefined) {
      return jsonResponse.data;
    }

    throw new Error(jsonResponse.message);
  }
}
```

---

## 📝 Notes for Future Reference

### Poster Image Display

All `poster_path` values are relative paths from TMDB. To display:

```typescript
// Transform in frontend
const fullPosterUrl = movie.poster_path
  ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
  : null;
```

### Genre Format Handling

Backend always returns genres as arrays, but parquet files may have pipe-separated strings. The `_parse_genres()` utility handles both:

```python
# Backend parsing
def _parse_genres(genres_data):
    if isinstance(genres_data, str):
        if '|' in genres_data:
            return [g.strip() for g in genres_data.split('|')]
    return genres_data if isinstance(genres_data, list) else []
```

### Response Unwrapping

Frontend automatically unwraps responses in `api.ts`:

```typescript
if (jsonResponse.success && jsonResponse.data !== undefined) {
  return jsonResponse.data;  // Return unwrapped data
}
```

This means frontend code works with `SearchResponse` directly, not `{success, data, message}`.

---

*Last Updated: 2025-01-04*
*For implementation details, see: docs/backend-frontend/backend-integration.md*
*For semantic search internals, see: docs/machine-learning/semantic-search.md*
