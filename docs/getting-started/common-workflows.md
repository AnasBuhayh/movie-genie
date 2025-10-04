# Common Workflows

Step-by-step guides for common tasks in Movie Genie development. Each workflow includes the exact commands to run, what to expect, and how to verify success.

---

## Table of Contents

1. [Changing Embedding Models](#changing-embedding-models)
2. [Adding New API Endpoints](#adding-new-api-endpoints)
3. [Debugging Search Issues](#debugging-search-issues)
4. [Regenerating Data with DVC](#regenerating-data-with-dvc)
5. [Testing the Full Stack](#testing-the-full-stack)
6. [Deploying Frontend Changes](#deploying-frontend-changes)

---

## Changing Embedding Models

**When**: You want to try a different embedding model for semantic search

**Time**: ~30 minutes (depending on dataset size)

### Step 1: Update Configuration Files

Both configs must use the **same model name**:

```bash
# 1. Update semantic search config
vim configs/semantic_search.yaml
```

Change:
```yaml
model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Your new model
```

```bash
# 2. Update data processing config
vim configs/data.yaml
```

Change:
```yaml
processing:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"  # MUST MATCH above
```

### Step 2: Verify Configuration

```bash
# Check both configs have the same model
grep model_name configs/semantic_search.yaml
grep embedding_model configs/data.yaml

# Output should be identical
```

### Step 3: Regenerate Embeddings

```bash
# This will:
# 1. Download the new model (if not cached)
# 2. Re-embed all 9,742 movies
# 3. Save to content_features.parquet

dvc repro content_features
```

**Expected output**:
```
Running stage 'content_features':
> python scripts/extract_content_features.py
...
INFO - Using embedding model: sentence-transformers/all-MiniLM-L6-v2
INFO - Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
INFO - Generating text embeddings...
Generating embeddings: 100%|████████| 9742/9742 [02:15<00:00, 71.92it/s]
INFO - Saved 9,742 movies with content features
```

**Time**: 2-5 minutes for ml-latest-small, longer for ml-25m

### Step 4: Verify Embeddings

```bash
# Check that embeddings were generated
python3 -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/content_features.parquet')
emb = np.array(df['text_embedding'].iloc[0])

print(f'✓ Movies: {len(df)}')
print(f'✓ Embedding dimension: {emb.shape}')
print(f'✓ Sample embedding norm: {np.linalg.norm(emb):.2f}')
"
```

**Expected output**:
```
✓ Movies: 9742
✓ Embedding dimension: (384,)
✓ Sample embedding norm: 12.34
```

### Step 5: Restart Backend

```bash
# Restart backend to reload search engine with new embeddings
dvc repro backend_server
```

**Check logs for**:
```
INFO - Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
INFO - Loaded 9742 total movie records
INFO - Extracted 9742 valid movie embeddings
INFO - Embedding dimension: 384
✅ SemanticSearchEngine initialized successfully
```

### Step 6: Test Search Quality

```bash
# Test search with a known query
curl "http://127.0.0.1:5001/api/search/semantic?q=batman&k=5" | jq '.data.movies[] | {title, similarity_score}'
```

**Expected**: Batman movies with similarity scores > 0.5

### Common Issues

**Issue**: "No valid text embeddings found"
- **Cause**: Dimension mismatch or embeddings not generated
- **Fix**: Check embedding dimension in Step 4, regenerate if needed

**Issue**: Search returns same results as before
- **Cause**: Backend not restarted or using cached search engine
- **Fix**: Kill backend process and run `dvc repro backend_server` again

---

## Adding New API Endpoints

**When**: You want to add a new feature that requires a new API endpoint

**Example**: Adding a "Get Movie Cast" endpoint

### Step 1: Define Backend Route

```bash
# Create or edit the appropriate API file
vim movie_genie/backend/app/api/movies.py
```

Add the endpoint:
```python
@movies_bp.route('/<int:movie_id>/cast', methods=['GET'])
def get_movie_cast(movie_id):
    """Get cast information for a movie"""
    try:
        # Load movie data
        df = pd.read_parquet('data/processed/content_features.parquet')
        movie = df[df['movieId'] == movie_id]

        if movie.empty:
            return jsonify({
                'success': False,
                'message': f'Movie {movie_id} not found'
            }), 404

        cast = movie.iloc[0].get('cast', [])

        return jsonify({
            'success': True,
            'message': 'Cast retrieved successfully',
            'data': {
                'movie_id': movie_id,
                'cast': cast
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get cast: {str(e)}'
        }), 500
```

### Step 2: Add to Frontend API Client

```bash
vim movie_genie/frontend/src/lib/api.ts
```

Add the endpoint constant:
```typescript
// In API_ENDPOINTS object
MOVIE_CAST: '/movies',
```

Add the method:
```typescript
// Get movie cast
static async getMovieCast(movieId: number): Promise<{movie_id: number, cast: string[]}> {
  const url = `${API_ENDPOINTS.MOVIE_CAST}/${movieId}/cast`;
  return this.fetchAPI(url);
}
```

### Step 3: Add to Service Layer (Optional)

```bash
vim movie_genie/frontend/src/services/movieDataService.ts
```

Add service method:
```typescript
static async getMovieCast(movieId: string): Promise<string[]> {
  try {
    const response = await MovieGenieAPI.getMovieCast(parseInt(movieId));
    return response.cast || [];
  } catch (error) {
    console.error('Failed to get cast:', error);
    return [];
  }
}
```

### Step 4: Update API Documentation

```bash
vim docs/backend-frontend/api-reference.md
```

Add endpoint documentation following the existing pattern.

### Step 5: Test the Endpoint

```bash
# 1. Restart backend
dvc repro backend_server

# 2. Test with curl
curl http://127.0.0.1:5001/api/movies/1/cast

# 3. Expected response
{
  "success": true,
  "message": "Cast retrieved successfully",
  "data": {
    "movie_id": 1,
    "cast": ["Tom Hanks", "Tim Allen"]
  }
}
```

### Step 6: Test from Frontend

```typescript
// In your component
const cast = await MovieDataService.getMovieCast('1');
console.log('Cast:', cast);
```

---

## Debugging Search Issues

**When**: Search returns unexpected results or errors

### Quick Diagnostic Steps

```bash
# 1. Check if backend is running
curl http://127.0.0.1:5001/api/health

# 2. Check search engine status
curl http://127.0.0.1:5001/api/search/status

# 3. Test search directly
curl "http://127.0.0.1:5001/api/search/semantic?q=test&k=5"
```

### Detailed Debugging Workflow

#### Step 1: Check Search Engine Initialization

```bash
# Look for initialization logs
tail -50 logs/backend.log | grep -i "search"

# Should see:
# ✅ SemanticSearchEngine initialized successfully
# Total movies indexed: 9742
# Embedding dimension: 384
```

#### Step 2: Test Embeddings Directly

```python
# Run this Python script
python3 << 'EOF'
from movie_genie.search.semantic_engine import SemanticSearchEngine
import logging
logging.basicConfig(level=logging.INFO)

# Initialize engine
engine = SemanticSearchEngine('configs/semantic_search.yaml')

# Test search
results = engine.search('batman', k=5)

print("\nSearch Results:")
for i, movie in enumerate(results, 1):
    print(f"{i}. {movie['title']}")
    print(f"   Similarity: {movie.get('similarity_score', 0):.4f}")
    print(f"   Genres: {movie.get('genres', [])}")
EOF
```

#### Step 3: Check Embedding Quality

```python
# Test if embeddings make sense
python3 << 'EOF'
from movie_genie.data.embeddings import TextEmbedder
import numpy as np

embedder = TextEmbedder("sentence-transformers/all-MiniLM-L6-v2")

# Encode related queries
batman = embedder.embed_texts(["batman"])[0]
superhero = embedder.embed_texts(["superhero movie"])[0]
comedy = embedder.embed_texts(["funny comedy"])[0]

# Normalize
batman_norm = batman / np.linalg.norm(batman)
superhero_norm = superhero / np.linalg.norm(superhero)
comedy_norm = comedy / np.linalg.norm(comedy)

# Compute similarities
print(f"batman vs superhero: {np.dot(batman_norm, superhero_norm):.4f}")  # Should be high
print(f"batman vs comedy: {np.dot(batman_norm, comedy_norm):.4f}")        # Should be low
EOF
```

#### Step 4: Check for Common Issues

```bash
# Issue: Wrong embedding dimension
python3 -c "
import pandas as pd
import numpy as np
df = pd.read_parquet('data/processed/content_features.parquet')
emb = np.array(df['text_embedding'].iloc[0])
print(f'Dimension: {emb.shape}')  # Should be (384,)
"

# Issue: Missing embeddings
python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/content_features.parquet')
has_emb = df['text_embedding'].notna().sum()
print(f'Movies with embeddings: {has_emb} / {len(df)}')  # Should be same
"

# Issue: Config mismatch
grep model_name configs/semantic_search.yaml
grep embedding_model configs/data.yaml
# Should match!
```

#### Step 5: Enable Debug Logging

```bash
# Edit backend app.py temporarily
vim movie_genie/backend/app.py
```

Change:
```python
logging.basicConfig(level=logging.DEBUG)  # Was INFO
```

Restart and check logs for detailed info.

---

## Regenerating Data with DVC

**When**: After code changes, data updates, or config changes

### Understanding DVC Pipeline

```bash
# View the pipeline
dvc dag

# Output shows dependency graph:
# ingest → sequential_processing → content_features → models → evaluation
```

### Regenerate Specific Stages

```bash
# Regenerate just content features
dvc repro content_features

# Regenerate from sequential processing onwards
dvc repro sequential_processing

# Regenerate everything
dvc repro
```

### Force Regeneration (Ignore Cache)

```bash
# Force re-run even if inputs haven't changed
dvc repro --force content_features

# Or set in config
vim configs/data.yaml
# Set: force_reprocess: true
```

### Check What Will Run

```bash
# See what stages are out of date
dvc status

# See what would run without actually running
dvc repro --dry
```

### Common DVC Workflows

#### Workflow 1: Changed Data Processing Code

```bash
# 1. Make code changes
vim movie_genie/data/processors.py

# 2. Mark stage as changed
dvc repro --force content_features

# 3. Downstream stages will auto-run
```

#### Workflow 2: Added New Raw Data

```bash
# 1. Add new data file
cp new_data.csv data/raw/

# 2. Update DVC tracking
dvc add data/raw/new_data.csv

# 3. Regenerate from ingest
dvc repro ingest
```

#### Workflow 3: Changed Model Hyperparameters

```bash
# 1. Update config
vim configs/bert4rec.yaml

# 2. Regenerate model
dvc repro bert4rec_training

# 3. Backend will auto-reload new model
```

---

## Testing the Full Stack

**When**: After major changes, before deployment

### End-to-End Test Workflow

#### Step 1: Start Backend

```bash
# In one terminal
dvc repro backend_server

# Wait for:
# ✅ SemanticSearchEngine initialized successfully
# * Running on http://127.0.0.1:5001
```

#### Step 2: Start Frontend

```bash
# In another terminal
cd movie_genie/frontend
npm run dev

# Wait for:
# Local: http://localhost:5173/
```

#### Step 3: Test API Endpoints

```bash
# Health check
curl http://127.0.0.1:5001/api/health

# User info
curl http://127.0.0.1:5001/api/users/info

# Popular movies
curl "http://127.0.0.1:5001/api/movies/popular?limit=5"

# Search
curl "http://127.0.0.1:5001/api/search/semantic?q=action&k=5"

# User profile
curl http://127.0.0.1:5001/api/users/1/profile

# Watched movies
curl http://127.0.0.1:5001/api/users/1/watched

# Historical interest
curl http://127.0.0.1:5001/api/users/1/historical-interest
```

#### Step 4: Test Frontend UI

1. Open http://localhost:5173/
2. Click "Change User" and enter user ID `1`
3. Wait for data to load
4. Verify:
   - Popular Movies section has real movie posters
   - "You Might Like These" section has different movies than Popular
   - "Based on Your Taste" section has movies
5. Search for "batman"
6. Verify:
   - Search results show Batman movies
   - Posters are displayed
   - No "Loading..." placeholders persist
7. Click on a movie
8. Verify movie details display correctly

#### Step 5: Check Browser Console

```
# Should see:
✅ Got real recommendations: 10
✅ Got real search results: 5
✅ Found 5 movies for "batman"

# Should NOT see:
⚠️ Using mock data
❌ Search failed
```

---

## Deploying Frontend Changes

**When**: After making frontend code changes

### Development Workflow

```bash
# Frontend auto-reloads with npm run dev
cd movie_genie/frontend
npm run dev

# Make changes, see them live at http://localhost:5173/
```

### Building for Production

```bash
# Build frontend assets
cd movie_genie/frontend
npm run build

# Output goes to dist/
ls -la dist/
```

### Deploying to Backend (Integrated Build)

```bash
# Use DVC to build and deploy frontend
dvc repro frontend_build

# This will:
# 1. Run npm run build
# 2. Copy index.html to backend/templates/
# 3. Copy assets to backend/static/

# Verify
ls -la movie_genie/backend/templates/index.html
ls -la movie_genie/backend/static/
```

### Testing Production Build

```bash
# Start backend (serves frontend from static/)
dvc repro backend_server

# Open in browser
open http://127.0.0.1:5001/

# Should see production frontend
```

### Common Frontend Issues

**Issue**: Changes not showing
- **Cause**: Browser cache
- **Fix**: Hard refresh (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)

**Issue**: Import errors after deployment
- **Cause**: Missing dependencies
- **Fix**: `npm install` and rebuild

**Issue**: Environment variables not working
- **Cause**: Using `process.env` instead of `import.meta.env`
- **Fix**: Change to `import.meta.env.VITE_*`

---

## Quick Reference Commands

```bash
# Development
dvc repro backend_server           # Start backend
cd frontend && npm run dev          # Start frontend dev server

# Data
dvc repro content_features          # Regenerate embeddings
dvc repro --force content_features  # Force regeneration

# Testing
curl http://127.0.0.1:5001/api/health                           # Health check
curl "http://127.0.0.1:5001/api/search/semantic?q=test&k=5"    # Test search
curl http://127.0.0.1:5001/api/search/status                    # Search engine status

# Building
dvc repro frontend_build            # Build and deploy frontend
dvc repro                           # Regenerate entire pipeline

# Debugging
tail -f logs/backend.log            # Watch backend logs
dvc status                          # Check what's out of date
grep model_name configs/*.yaml      # Verify config consistency
```

---

*Last Updated: January 2025*
*See also: docs/reference/configuration.md for config details*
