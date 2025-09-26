"""
Recommendation Service - Integrates BERT4Rec and Two-Tower models

This service provides personalized movie recommendations using your trained models.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service for generating personalized movie recommendations"""

    def __init__(self,
                 bert4rec_model_path: Optional[str] = None,
                 bert4rec_artifacts_path: Optional[str] = None,
                 two_tower_model_path: Optional[str] = None):
        """
        Initialize recommendation service with ML models

        Args:
            bert4rec_model_path: Path to BERT4Rec model file
            bert4rec_artifacts_path: Path to BERT4Rec data artifacts
            two_tower_model_path: Path to Two-Tower model file
        """
        # Set default paths
        models_dir = project_root / "models"
        self.bert4rec_model_path = bert4rec_model_path or str(models_dir / "bert4rec" / "bert4rec_model.pth")
        self.bert4rec_artifacts_path = bert4rec_artifacts_path or str(models_dir / "bert4rec" / "data_artifacts.pkl")
        self.two_tower_model_path = two_tower_model_path or str(models_dir / "two_tower" / "two_tower_model.pth")

        # Initialize models
        self.bert4rec_reranker = None
        self.two_tower_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for recommendations"""
        try:
            from movie_genie.search.rerankers import BERT4RecReranker, TwoTowerReranker

            # Initialize BERT4Rec reranker
            logger.info("Initializing BERT4Rec reranker...")
            self.bert4rec_reranker = BERT4RecReranker(
                self.bert4rec_model_path,
                self.bert4rec_artifacts_path
            )

            # Initialize Two-Tower reranker
            logger.info("Initializing Two-Tower reranker...")
            self.two_tower_reranker = TwoTowerReranker(self.two_tower_model_path)

            logger.info("✅ Recommendation models initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize recommendation models: {e}")
            self.bert4rec_reranker = None
            self.two_tower_reranker = None

    def get_personalized_recommendations(self,
                                       user_id: Optional[str] = None,
                                       interaction_history: Optional[List[Dict]] = None,
                                       limit: int = 20) -> Dict[str, Any]:
        """
        Get personalized recommendations for a user

        Args:
            user_id: User ID for recommendations
            interaction_history: User's movie interaction history
            limit: Number of recommendations to return

        Returns:
            Dictionary with personalized movie recommendations
        """
        if not self.bert4rec_reranker:
            logger.error("BERT4Rec model not available")
            return self._fallback_recommendations(limit)

        try:
            logger.info(f"Generating personalized recommendations for user {user_id}")

            # First, get popular movies as candidates
            from .movie_service import MovieService
            movie_service = MovieService()
            candidate_movies = movie_service.get_popular_movies(limit * 3)  # Get more candidates for reranking

            if not candidate_movies:
                logger.warning("No candidate movies found")
                return self._fallback_recommendations(limit)

            # Format candidates for reranker
            search_results = []
            for movie in candidate_movies:
                search_results.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'overview': movie['overview'],
                    'genres': movie['genres'],
                    'similarity_score': 1.0,  # Default similarity score
                    'rank': len(search_results) + 1
                })

            # Create user context for reranking
            user_context = {}
            if user_id:
                user_context['user_id'] = user_id
            if interaction_history:
                user_context['interaction_history'] = interaction_history

            # Apply BERT4Rec reranking
            if user_context:
                logger.info("Applying BERT4Rec personalized reranking...")
                reranked_results = self.bert4rec_reranker.rerank(search_results, user_context)
            else:
                reranked_results = search_results

            # Take top results
            recommendations = reranked_results[:limit]

            response = {
                'movies': recommendations,
                'recommendation_type': 'personalized',
                'model': 'BERT4Rec',
                'total': len(recommendations),
                'user_context': user_context
            }

            logger.info(f"✅ Generated {len(recommendations)} personalized recommendations")
            return response

        except Exception as e:
            logger.error(f"❌ Error generating personalized recommendations: {e}")
            return self._fallback_recommendations(limit)

    def get_content_based_recommendations(self,
                                        movie_id: int,
                                        limit: int = 10) -> Dict[str, Any]:
        """
        Get content-based recommendations for a specific movie

        Args:
            movie_id: Movie ID to base recommendations on
            limit: Number of recommendations to return

        Returns:
            Dictionary with content-based recommendations
        """
        try:
            logger.info(f"Generating content-based recommendations for movie {movie_id}")

            # Use semantic search engine to find similar movies
            from .search_service import SearchService
            from .movie_service import MovieService

            movie_service = MovieService()
            search_service = SearchService()

            # Get the source movie details
            source_movie = movie_service.get_movie_details(movie_id)
            if not source_movie:
                logger.warning(f"Source movie {movie_id} not found")
                return {'movies': [], 'recommendation_type': 'content_based', 'total': 0}

            # Create a search query from the movie's content
            query_parts = []
            if source_movie.get('title'):
                query_parts.append(source_movie['title'])
            if source_movie.get('genres'):
                query_parts.extend(source_movie['genres'])
            if source_movie.get('overview'):
                # Take first few words of overview
                overview_words = source_movie['overview'].split()[:10]
                query_parts.extend(overview_words)

            search_query = ' '.join(query_parts)

            # Perform semantic search
            similar_movies = search_service.semantic_search(search_query, limit + 5)

            # Filter out the source movie itself
            recommendations = [
                movie for movie in similar_movies.get('movies', [])
                if movie.get('movieId') != movie_id
            ][:limit]

            response = {
                'movies': recommendations,
                'recommendation_type': 'content_based',
                'source_movie': source_movie,
                'total': len(recommendations)
            }

            logger.info(f"✅ Generated {len(recommendations)} content-based recommendations")
            return response

        except Exception as e:
            logger.error(f"❌ Error generating content-based recommendations: {e}")
            return {'movies': [], 'recommendation_type': 'content_based', 'total': 0}

    def get_collaborative_recommendations(self,
                                        user_id: str,
                                        limit: int = 20) -> Dict[str, Any]:
        """
        Get collaborative filtering recommendations using Two-Tower model

        Args:
            user_id: User ID for recommendations
            limit: Number of recommendations to return

        Returns:
            Dictionary with collaborative recommendations
        """
        if not self.two_tower_reranker:
            logger.error("Two-Tower model not available")
            return self._fallback_recommendations(limit)

        try:
            logger.info(f"Generating collaborative recommendations for user {user_id}")

            # Get candidate movies
            from .movie_service import MovieService
            movie_service = MovieService()
            candidate_movies = movie_service.get_popular_movies(limit * 2)

            # Format for reranking
            search_results = []
            for movie in candidate_movies:
                search_results.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'overview': movie['overview'],
                    'genres': movie['genres'],
                    'similarity_score': 1.0,
                    'rank': len(search_results) + 1
                })

            # Apply Two-Tower reranking
            user_context = {'user_id': user_id}
            reranked_results = self.two_tower_reranker.rerank(search_results, user_context)

            recommendations = reranked_results[:limit]

            response = {
                'movies': recommendations,
                'recommendation_type': 'collaborative',
                'model': 'Two-Tower',
                'total': len(recommendations)
            }

            logger.info(f"✅ Generated {len(recommendations)} collaborative recommendations")
            return response

        except Exception as e:
            logger.error(f"❌ Error generating collaborative recommendations: {e}")
            return self._fallback_recommendations(limit)

    def _fallback_recommendations(self, limit: int) -> Dict[str, Any]:
        """
        Fallback to popular movies when personalized recommendations fail

        Args:
            limit: Number of recommendations to return

        Returns:
            Dictionary with popular movies as fallback
        """
        try:
            logger.info("Using fallback recommendations (popular movies)")

            from .movie_service import MovieService
            movie_service = MovieService()
            popular_movies = movie_service.get_popular_movies(limit)

            return {
                'movies': popular_movies,
                'recommendation_type': 'popular',
                'total': len(popular_movies),
                'note': 'Fallback to popular movies due to model unavailability'
            }

        except Exception as e:
            logger.error(f"❌ Even fallback recommendations failed: {e}")
            return {
                'movies': [],
                'recommendation_type': 'popular',
                'total': 0,
                'error': str(e)
            }

    def is_available(self) -> bool:
        """Check if recommendation service is available"""
        return self.bert4rec_reranker is not None or self.two_tower_reranker is not None

    def get_status(self) -> Dict[str, Any]:
        """Get recommendation service status information"""
        return {
            'bert4rec_available': self.bert4rec_reranker is not None,
            'two_tower_available': self.two_tower_reranker is not None,
            'bert4rec_model_path': self.bert4rec_model_path,
            'two_tower_model_path': self.two_tower_model_path,
            'service': 'RecommendationService'
        }