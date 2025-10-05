# How to Serve Recommendations via API

This guide shows you how to create API endpoints to serve recommendations from your trained models.

## Prerequisites

- Trained model saved in `models/` directory
- Flask backend set up (see [Backend Integration](../backend-frontend/backend-integration.md))
- Understanding of Flask blueprints and REST APIs

## Overview

Serving recommendations involves:

1. Loading your trained model in the backend
2. Creating a model service class
3. Adding Flask API routes
4. Implementing recommendation logic
5. Handling errors and edge cases
6. Testing the endpoint
7. Connecting to the frontend

## Step 1: Create Model Service

Create a service class to manage your model:

```python
# movie_genie/backend/app/services/your_model_service.py

import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
import pandas as pd

from movie_genie.models.your_model import YourModel

logger = logging.getLogger(__name__)


class YourModelService:
    """Service for serving recommendations from YourModel."""

    def __init__(self, model_path: str, data_path: str):
        """
        Initialize the model service.

        Args:
            model_path: Path to saved model directory
            data_path: Path to processed data files
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.movie_metadata = None

        # Load everything
        self._load_model()
        self._load_mappings()
        self._load_metadata()

    def _load_model(self):
        """Load the trained model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = YourModel.from_pretrained(str(self.model_path))
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_mappings(self):
        """Load user and item ID mappings."""
        try:
            # Load mappings from saved files
            # Adjust paths based on your data structure
            import json

            mappings_path = self.model_path / "mappings.json"
            if mappings_path.exists():
                with open(mappings_path) as f:
                    mappings = json.load(f)

                self.user_to_idx = {int(k): v for k, v in mappings['user_to_idx'].items()}
                self.item_to_idx = {int(k): v for k, v in mappings['item_to_idx'].items()}
                self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
                self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}

                logger.info(f"Loaded {len(self.user_to_idx)} users, {len(self.item_to_idx)} items")

        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            raise

    def _load_metadata(self):
        """Load movie metadata for enriching recommendations."""
        try:
            metadata_path = self.data_path / "content_features.parquet"
            if metadata_path.exists():
                self.movie_metadata = pd.read_parquet(metadata_path)
                logger.info(f"Loaded metadata for {len(self.movie_metadata)} movies")
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            self.movie_metadata = None

    def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        return_scores: bool = False
    ) -> List[Dict]:
        """
        Get top-k recommendations for a user.

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude items user has already seen
            return_scores: Whether to include prediction scores

        Returns:
            List of recommended items with metadata
        """
        try:
            # Check if user exists
            if user_id not in self.user_to_idx:
                logger.warning(f"Unknown user: {user_id}")
                return self._get_popular_items(k)

            # Get user index
            user_idx = self.user_to_idx[user_id]

            # Get user's interaction history if excluding seen items
            seen_items = set()
            if exclude_seen:
                seen_items = self._get_user_history(user_id)

            # Score all items
            item_scores = self._score_all_items(user_idx)

            # Filter out seen items
            if seen_items:
                for item_id in seen_items:
                    if item_id in self.item_to_idx:
                        item_idx = self.item_to_idx[item_id]
                        item_scores[item_idx] = -float('inf')

            # Get top-k items
            top_k_indices = torch.topk(item_scores, k=min(k, len(item_scores))).indices.cpu().numpy()

            # Convert to item IDs and add metadata
            recommendations = []
            for idx in top_k_indices:
                item_id = self.idx_to_item[idx]
                rec = self._enrich_item(item_id)

                if return_scores:
                    rec['score'] = float(item_scores[idx])

                recommendations.append(rec)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            # Fallback to popular items
            return self._get_popular_items(k)

    def _score_all_items(self, user_idx: int) -> torch.Tensor:
        """
        Score all items for a given user.

        Args:
            user_idx: User index

        Returns:
            scores: Tensor of scores for all items
        """
        num_items = len(self.item_to_idx)

        # Create batch of (user_idx, item_idx) pairs
        user_indices = torch.full((num_items,), user_idx, dtype=torch.long)
        item_indices = torch.arange(num_items, dtype=torch.long)

        # Move to device
        user_indices = user_indices.to(self.device)
        item_indices = item_indices.to(self.device)

        # Get predictions
        with torch.no_grad():
            scores = self.model.predict_batch(user_indices, item_indices, self.device)

        return scores.cpu()

    def _get_user_history(self, user_id: int) -> set:
        """
        Get items the user has already interacted with.

        Args:
            user_id: User ID

        Returns:
            Set of item IDs user has seen
        """
        # Load from sequences or interactions data
        # This is a placeholder - implement based on your data structure
        sequences_path = self.data_path / "sequences_with_metadata.parquet"

        if sequences_path.exists():
            sequences = pd.read_parquet(sequences_path)
            user_sequences = sequences[sequences['user_id'] == user_id]

            if not user_sequences.empty:
                # Get all items from user's sequence
                seen_items = set()
                for seq in user_sequences['sequence'].values:
                    seen_items.update(seq)
                return seen_items

        return set()

    def _enrich_item(self, item_id: int) -> Dict:
        """
        Add metadata to item.

        Args:
            item_id: Item ID

        Returns:
            Dict with item info and metadata
        """
        result = {'movie_id': item_id}

        if self.movie_metadata is not None:
            movie_data = self.movie_metadata[
                self.movie_metadata['movie_id'] == item_id
            ]

            if not movie_data.empty:
                movie = movie_data.iloc[0]
                result.update({
                    'title': movie.get('title', f'Movie {item_id}'),
                    'genres': movie.get('genres', []),
                    'year': movie.get('year'),
                    'tmdb_id': movie.get('tmdb_id'),
                    'poster_path': movie.get('poster_path'),
                    'overview': movie.get('overview'),
                })
        else:
            result['title'] = f'Movie {item_id}'

        return result

    def _get_popular_items(self, k: int) -> List[Dict]:
        """
        Fallback: return popular items.

        Args:
            k: Number of items to return

        Returns:
            List of popular items
        """
        logger.info("Returning popular items as fallback")

        # Load popular items from data or use a cached list
        # This is a placeholder
        popular_ids = list(self.item_to_idx.keys())[:k]

        return [self._enrich_item(item_id) for item_id in popular_ids]

    def get_similar_items(
        self,
        item_id: int,
        k: int = 10
    ) -> List[Dict]:
        """
        Get items similar to a given item.

        Args:
            item_id: Reference item ID
            k: Number of similar items

        Returns:
            List of similar items
        """
        try:
            if item_id not in self.item_to_idx:
                logger.warning(f"Unknown item: {item_id}")
                return []

            item_idx = self.item_to_idx[item_id]

            # Get item embedding
            item_embedding = self.model.get_item_embedding(item_idx)

            # Compute similarity with all items
            all_item_embeddings = self.model.item_embeddings.weight
            similarities = torch.cosine_similarity(
                item_embedding.unsqueeze(0),
                all_item_embeddings,
                dim=1
            )

            # Exclude the item itself
            similarities[item_idx] = -float('inf')

            # Get top-k
            top_k_indices = torch.topk(similarities, k=min(k, len(similarities))).indices.cpu().numpy()

            # Convert to items with metadata
            similar_items = []
            for idx in top_k_indices:
                similar_id = self.idx_to_item[idx]
                item = self._enrich_item(similar_id)
                item['similarity'] = float(similarities[idx])
                similar_items.append(item)

            return similar_items

        except Exception as e:
            logger.error(f"Error getting similar items for {item_id}: {e}")
            return []
```

## Step 2: Register Service in Application

Update the backend initialization to load your model service:

```python
# movie_genie/backend/app/__init__.py

from flask import Flask
from movie_genie.backend.app.services.your_model_service import YourModelService

# Global service instance
your_model_service = None


def create_app():
    app = Flask(__name__)

    # Load model services
    global your_model_service

    try:
        your_model_service = YourModelService(
            model_path="models/your_model",
            data_path="data/processed"
        )
        app.logger.info("YourModel service loaded successfully")
    except Exception as e:
        app.logger.error(f"Failed to load YourModel service: {e}")
        your_model_service = None

    # Register blueprints
    from movie_genie.backend.app.api import recommendations_routes
    app.register_blueprint(recommendations_routes.bp)

    return app
```

## Step 3: Create API Routes

Add routes for your recommendation endpoint:

```python
# movie_genie/backend/app/api/recommendations_routes.py

from flask import Blueprint, request, jsonify
import logging

from movie_genie.backend.app import your_model_service

logger = logging.getLogger(__name__)

bp = Blueprint('recommendations', __name__, url_prefix='/api/recommendations')


@bp.route('/your-model', methods=['POST'])
def get_your_model_recommendations():
    """
    Get recommendations from YourModel.

    Request body:
    {
        "user_id": int,
        "k": int (optional, default 10),
        "exclude_seen": bool (optional, default true),
        "return_scores": bool (optional, default false)
    }

    Returns:
    {
        "success": bool,
        "user_id": int,
        "recommendations": [
            {
                "movie_id": int,
                "title": str,
                "genres": list,
                "year": int,
                "score": float (if return_scores=true)
            }
        ],
        "count": int
    }
    """
    try:
        # Parse request
        data = request.get_json()

        if not data or 'user_id' not in data:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400

        user_id = int(data['user_id'])
        k = int(data.get('k', 10))
        exclude_seen = data.get('exclude_seen', True)
        return_scores = data.get('return_scores', False)

        # Validate parameters
        if k < 1 or k > 100:
            return jsonify({
                'success': False,
                'error': 'k must be between 1 and 100'
            }), 400

        # Check if service is available
        if your_model_service is None:
            return jsonify({
                'success': False,
                'error': 'YourModel service not available'
            }), 503

        # Get recommendations
        recommendations = your_model_service.get_recommendations(
            user_id=user_id,
            k=k,
            exclude_seen=exclude_seen,
            return_scores=return_scores
        )

        return jsonify({
            'success': True,
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations),
            'model': 'your-model'
        })

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@bp.route('/similar-items', methods=['POST'])
def get_similar_items():
    """
    Get items similar to a given item.

    Request body:
    {
        "item_id": int,
        "k": int (optional, default 10)
    }

    Returns:
    {
        "success": bool,
        "item_id": int,
        "similar_items": [
            {
                "movie_id": int,
                "title": str,
                "similarity": float
            }
        ],
        "count": int
    }
    """
    try:
        data = request.get_json()

        if not data or 'item_id' not in data:
            return jsonify({
                'success': False,
                'error': 'item_id is required'
            }), 400

        item_id = int(data['item_id'])
        k = int(data.get('k', 10))

        if your_model_service is None:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 503

        similar_items = your_model_service.get_similar_items(
            item_id=item_id,
            k=k
        )

        return jsonify({
            'success': True,
            'item_id': item_id,
            'similar_items': similar_items,
            'count': len(similar_items)
        })

    except Exception as e:
        logger.error(f"Error getting similar items: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

## Step 4: Test Your Endpoint

### Using curl

```bash
# Start the backend
FLASK_PORT=5001 python scripts/start_server.py

# Test recommendations endpoint
curl -X POST http://localhost:5001/api/recommendations/your-model \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "k": 10,
    "exclude_seen": true,
    "return_scores": true
  }' | jq

# Test similar items endpoint
curl -X POST http://localhost:5001/api/recommendations/similar-items \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": 456,
    "k": 5
  }' | jq
```

### Using Python

```python
import requests

# Get recommendations
response = requests.post(
    'http://localhost:5001/api/recommendations/your-model',
    json={
        'user_id': 123,
        'k': 10,
        'exclude_seen': True
    }
)

print(response.json())

# Expected output:
# {
#   'success': True,
#   'user_id': 123,
#   'recommendations': [
#     {
#       'movie_id': 789,
#       'title': 'The Matrix',
#       'genres': ['Action', 'Sci-Fi'],
#       'year': 1999
#     },
#     ...
#   ],
#   'count': 10
# }
```

## Step 5: Add Frontend Integration

Create a frontend service to call your API:

```typescript
// movie_genie/frontend/src/services/yourModelService.ts

import { api } from './api';

export interface Recommendation {
  movie_id: number;
  title: string;
  genres: string[];
  year?: number;
  score?: number;
  poster_path?: string;
}

export interface RecommendationsResponse {
  success: boolean;
  user_id: number;
  recommendations: Recommendation[];
  count: number;
  model: string;
}

export async function getYourModelRecommendations(
  userId: number,
  k: number = 10,
  excludeSeen: boolean = true,
  returnScores: boolean = false
): Promise<Recommendation[]> {
  const response = await api.post<RecommendationsResponse>(
    '/recommendations/your-model',
    {
      user_id: userId,
      k,
      exclude_seen: excludeSeen,
      return_scores: returnScores
    }
  );

  if (!response.data.success) {
    throw new Error('Failed to get recommendations');
  }

  return response.data.recommendations;
}

export async function getSimilarItems(
  itemId: number,
  k: number = 10
): Promise<Recommendation[]> {
  const response = await api.post<RecommendationsResponse>(
    '/recommendations/similar-items',
    {
      item_id: itemId,
      k
    }
  );

  if (!response.data.success) {
    throw new Error('Failed to get similar items');
  }

  return response.data.similar_items;
}
```

Create a React component to use it:

```tsx
// movie_genie/frontend/src/components/YourModelRecommendations.tsx

import { useState, useEffect } from 'react';
import { getYourModelRecommendations } from '@/services/yourModelService';
import type { Recommendation } from '@/services/yourModelService';

interface Props {
  userId: number;
  limit?: number;
}

export function YourModelRecommendations({ userId, limit = 10 }: Props) {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchRecommendations() {
      try {
        setLoading(true);
        const recs = await getYourModelRecommendations(userId, limit);
        setRecommendations(recs);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    fetchRecommendations();
  }, [userId, limit]);

  if (loading) return <div>Loading recommendations...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="recommendations-grid">
      {recommendations.map(rec => (
        <div key={rec.movie_id} className="recommendation-card">
          <h3>{rec.title}</h3>
          <p>{rec.genres.join(', ')}</p>
          {rec.year && <p>{rec.year}</p>}
          {rec.score && <p>Score: {rec.score.toFixed(3)}</p>}
        </div>
      ))}
    </div>
  );
}
```

## Step 6: Add Error Handling

Implement comprehensive error handling:

```python
# In your service

class YourModelServiceError(Exception):
    """Custom exception for model service errors."""
    pass

class ModelNotLoadedError(YourModelServiceError):
    """Model failed to load."""
    pass

class UserNotFoundError(YourModelServiceError):
    """User ID not found."""
    pass

# In your routes

@bp.errorhandler(ModelNotLoadedError)
def handle_model_not_loaded(e):
    return jsonify({
        'success': False,
        'error': 'Model service unavailable',
        'code': 'MODEL_NOT_LOADED'
    }), 503

@bp.errorhandler(UserNotFoundError)
def handle_user_not_found(e):
    return jsonify({
        'success': False,
        'error': 'User not found',
        'code': 'USER_NOT_FOUND'
    }), 404
```

## Step 7: Add Caching (Optional)

For better performance, add caching:

```python
from functools import lru_cache
from flask_caching import Cache

cache = Cache(config={'CACHE_TYPE': 'simple'})

# In your service
@cache.memoize(timeout=300)  # Cache for 5 minutes
def get_recommendations(self, user_id: int, k: int = 10):
    # ... implementation
    pass

# Clear cache when model is retrained
def clear_cache():
    cache.clear()
```

## Best Practices

### 1. Batch Processing

```python
def get_batch_recommendations(
    self,
    user_ids: List[int],
    k: int = 10
) -> Dict[int, List[Dict]]:
    """Get recommendations for multiple users efficiently."""
    results = {}
    for user_id in user_ids:
        results[user_id] = self.get_recommendations(user_id, k)
    return results
```

### 2. Pagination

```python
@bp.route('/your-model', methods=['POST'])
def get_recommendations_paginated():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))

    offset = (page - 1) * page_size

    # Get more recommendations and slice
    recs = your_model_service.get_recommendations(
        user_id=user_id,
        k=offset + page_size
    )

    paginated_recs = recs[offset:offset + page_size]

    return jsonify({
        'success': True,
        'recommendations': paginated_recs,
        'page': page,
        'page_size': page_size,
        'has_more': len(recs) > offset + page_size
    })
```

### 3. Request Logging

```python
import time

@bp.before_request
def log_request():
    request.start_time = time.time()

@bp.after_request
def log_response(response):
    duration = time.time() - request.start_time
    logger.info(
        f"{request.method} {request.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    return response
```

## Troubleshooting

### Service Not Loading

```bash
# Check if model files exist
ls -la models/your_model/

# Should see:
# - pytorch_model.bin
# - config.json
# - mappings.json

# Check logs
tail -f logs/backend.log
```

### Recommendations Empty

- Verify user exists in mappings
- Check if all items are being filtered out
- Test with `exclude_seen=False`

### Slow Response Times

- Add caching
- Use batch processing
- Profile with `cProfile`
- Consider using async/await

## Next Steps

- [Adding a New Model](add-new-model.md)
- [MLflow Integration](mlflow-integration.md)
- [API Reference](../backend-frontend/api-reference.md)
- [Frontend Development](../backend-frontend/index.md)

## Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [REST API Best Practices](https://restfulapi.net/)
- [PyTorch Inference](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
