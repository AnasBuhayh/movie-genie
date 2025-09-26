import numpy as np
import torch
import pickle
import logging
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add path for imports
sys.path.append('.')

logger = logging.getLogger(__name__)

class SearchReranker(ABC):
    """Abstract interface for reranking search results using recommendation models."""

    @abstractmethod
    def rerank(self, search_results: List[Dict[str, Any]],
               user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Rerank search results based on user preferences."""
        pass

class BERT4RecReranker(SearchReranker):
    """Rerank using BERT4Rec sequential recommendation model."""

    def __init__(self, model_path: str, data_artifacts_path: str):
        self.model_path = Path(model_path)
        self.data_artifacts_path = Path(data_artifacts_path)
        self.model = None
        self.data_artifacts = None
        self._load_model()

    def _load_model(self):
        """Load BERT4Rec model and data artifacts."""
        try:
            # Load data artifacts
            if self.data_artifacts_path.exists():
                with open(self.data_artifacts_path, 'rb') as f:
                    self.data_artifacts = pickle.load(f)
                logger.info(f"Loaded BERT4Rec data artifacts from {self.data_artifacts_path}")
            else:
                logger.warning(f"Data artifacts not found at {self.data_artifacts_path}")
                return

            # Load model state dict
            if self.model_path.exists():
                state_dict = torch.load(self.model_path, map_location='cpu')

                # Import the model class
                try:
                    from movie_genie.ranking.bert4rec_model import BERT4RecModel

                    # Create model with correct parameters
                    num_items = self.data_artifacts['num_movies']

                    # Determine content_feature_dim from the saved model
                    if 'content_projection.weight' in state_dict:
                        content_feature_dim = state_dict['content_projection.weight'].shape[1]
                    else:
                        content_feature_dim = 768  # Fallback to text embedding dimension

                    # Use standard BERT4Rec config parameters
                    self.model = BERT4RecModel(
                        num_items=num_items,
                        content_feature_dim=content_feature_dim,
                        max_seq_len=50,      # From config
                        hidden_dim=256,      # From config
                        num_layers=4,        # From config
                        num_heads=8,         # From config
                        dropout_rate=0.1     # From config
                    )

                    # Load the state dict
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    logger.info(f"Successfully loaded BERT4Rec model from {self.model_path}")

                except ImportError as e:
                    logger.error(f"Could not import BERT4RecModel: {e}")
                    self.model = None

            else:
                logger.warning(f"Model not found at {self.model_path}")
                self.model = None

        except Exception as e:
            logger.error(f"Failed to load BERT4Rec model: {e}")
            self.model = None
            self.data_artifacts = None

    def rerank(self, search_results: List[Dict[str, Any]],
               user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Rerank using BERT4Rec sequential modeling."""
        if not user_context or not user_context.get('interaction_history'):
            return search_results  # Return original order if no user history

        if self.model is None or self.data_artifacts is None:
            logger.warning("BERT4Rec model not available, returning original results")
            return search_results

        # Get candidate movie IDs from search results
        candidate_ids = [result['movieId'] for result in search_results]
        user_sequence = user_context['interaction_history']

        # Score candidates using BERT4Rec
        candidate_scores = self._score_candidates_with_bert4rec(
            candidate_ids, user_sequence
        )

        # Combine semantic similarity with personalization scores
        for i, result in enumerate(search_results):
            semantic_score = result['similarity_score']
            personal_score = candidate_scores.get(result['movieId'], 0.0)

            # Weighted combination (configurable)
            combined_score = (0.6 * semantic_score + 0.4 * personal_score)
            result['personalized_score'] = combined_score
            result['bert4rec_score'] = personal_score

        # Sort by combined score
        reranked = sorted(search_results,
                         key=lambda x: x['personalized_score'],
                         reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result['rank'] = i + 1

        return reranked

    def _score_candidates_with_bert4rec(self, candidate_ids: List[int],
                                      user_sequence: List[Dict]) -> Dict[int, float]:
        """Score candidates using BERT4Rec model."""
        # Simplified scoring - return dummy scores for now
        # In a real implementation, you would:
        # 1. Convert user_sequence to model input format
        # 2. Run model inference to get next-item predictions
        # 3. Extract scores for the candidate items

        scores = {}
        for movie_id in candidate_ids:
            # Return a small random score to demonstrate personalization
            scores[movie_id] = np.random.uniform(0.1, 0.9)
        return scores

class TwoTowerReranker(SearchReranker):
    """Rerank using two-tower recommendation model."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load two-tower model with proper model reconstruction."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}")
                return

            # Look for model configuration
            config_path = self.model_path.parent / 'model_config.json'
            if not config_path.exists():
                logger.warning(f"Model config not found at {config_path}")
                return

            # Load model configuration
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load the state dict
            state_dict = torch.load(self.model_path, map_location='cpu')

            # Check if it's already a complete model or just a state dict
            if hasattr(state_dict, 'eval'):
                # It's a complete model
                self.model = state_dict
                logger.info(f"Loaded complete Two-Tower model from {self.model_path}")
            else:
                # It's a state dict - need to reconstruct the model
                from movie_genie.retrieval.two_tower_model import TwoTowerModel

                # Extract parameters from config
                model_config = config.get('training_config', {}).get('model', config)
                self.model = TwoTowerModel(
                    num_users=config['num_users'],
                    num_movies=config['num_movies'],
                    content_feature_dim=config['content_feature_dim'],
                    embedding_dim=model_config.get('embedding_dim', 128),
                    user_hidden_dims=model_config.get('user_hidden_dims', [128, 64]),
                    item_hidden_dims=model_config.get('item_hidden_dims', [256, 128]),
                    dropout_rate=model_config.get('dropout_rate', 0.1)
                )

                # Load the state dict into the model
                self.model.load_state_dict(state_dict)
                logger.info(f"Reconstructed Two-Tower model from state dict at {self.model_path}")

            # Set model to evaluation mode
            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load Two-Tower model: {e}")
            self.model = None

    def rerank(self, search_results: List[Dict[str, Any]],
               user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Rerank using two-tower collaborative filtering."""
        if not user_context or not user_context.get('user_id'):
            return search_results

        if self.model is None:
            logger.warning("Two-Tower model not available, returning original results")
            return search_results

        user_id = user_context['user_id']
        candidate_ids = [result['movieId'] for result in search_results]

        # Get two-tower scores for candidates
        candidate_scores = self._score_with_two_tower(user_id, candidate_ids)

        # Combine scores and rerank
        for result in search_results:
            semantic_score = result['similarity_score']
            collab_score = candidate_scores.get(result['movieId'], 0.0)
            result['personalized_score'] = 0.7 * semantic_score + 0.3 * collab_score

        return sorted(search_results,
                     key=lambda x: x['personalized_score'],
                     reverse=True)

    def _score_with_two_tower(self, user_id: int, candidate_ids: List[int]) -> Dict[int, float]:
        """Score candidates using two-tower model."""
        # Simplified scoring - return dummy scores for now
        scores = {}
        for movie_id in candidate_ids:
            scores[movie_id] = np.random.uniform(0.1, 0.9)
        return scores

class NoOpReranker(SearchReranker):
    """Pass-through reranker that returns original search order."""

    def rerank(self, search_results: List[Dict[str, Any]],
               user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return search_results