"""
Search Service - Integrates SemanticSearchEngine with Flask API

This service provides a clean interface to your semantic search functionality.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class SearchService:
    """Service for handling movie search operations"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize search service with semantic search engine

        Args:
            config_path: Path to semantic search configuration file
        """
        self.semantic_engine = None
        self.config_path = config_path or str(project_root / "configs" / "semantic_search.yaml")
        self._initialize_search_engine()

    def _initialize_search_engine(self):
        """Initialize the semantic search engine"""
        try:
            from movie_genie.search.semantic_engine import SemanticSearchEngine

            logger.info("Initializing SemanticSearchEngine...")
            self.semantic_engine = SemanticSearchEngine(self.config_path)
            logger.info("✅ SemanticSearchEngine initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize SemanticSearchEngine: {e}")
            self.semantic_engine = None

    def semantic_search(self,
                       query: str,
                       k: int = 20,
                       user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform semantic search for movies

        Args:
            query: Search query string
            k: Number of results to return
            user_context: Optional user context for personalization

        Returns:
            Dictionary with search results and metadata
        """
        if not self.semantic_engine:
            logger.error("SemanticSearchEngine not available")
            return {
                'movies': [],
                'total': 0,
                'query': query,
                'search_type': 'semantic',
                'error': 'Search engine not available'
            }

        try:
            logger.info(f"Performing semantic search for: '{query}' (k={k})")

            # Use your semantic engine's search method
            results = self.semantic_engine.search(query, k, user_context)

            # Format response for API
            formatted_results = []
            for movie in results:
                formatted_movie = {
                    'movieId': movie.get('movieId'),
                    'title': movie.get('title'),
                    'overview': movie.get('overview'),
                    'genres': self._parse_genres(movie.get('genres', [])),
                    'similarity_score': movie.get('similarity_score'),
                    'personalized_score': movie.get('personalized_score'),
                    'rank': movie.get('rank'),
                    'vote_average': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count'),
                    'runtime': movie.get('runtime'),
                    'release_date': movie.get('release_date'),
                    'poster_path': movie.get('poster_path'),
                    'backdrop_path': movie.get('backdrop_path')
                }
                formatted_results.append(formatted_movie)

            response = {
                'movies': formatted_results,
                'total': len(formatted_results),
                'query': query,
                'search_type': 'semantic',
                'personalized': user_context is not None
            }

            logger.info(f"✅ Semantic search completed: {len(formatted_results)} results")
            return response

        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return {
                'movies': [],
                'total': 0,
                'query': query,
                'search_type': 'semantic',
                'error': str(e)
            }

    def traditional_search(self, query: str, k: int = 20) -> Dict[str, Any]:
        """
        Perform traditional title-based search (fallback when semantic search fails)

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        logger.info(f"Performing traditional title-based search for: '{query}' (k={k})")

        try:
            # Use movie service for title-based search
            from movie_genie.backend.app.services.movie_service import MovieService
            movie_service = MovieService()

            movies = movie_service.search_movies_by_title(query, k)

            # Format response to match semantic search structure
            formatted_results = []
            for idx, movie in enumerate(movies):
                formatted_movie = {
                    'movieId': movie.get('movieId'),
                    'title': movie.get('title'),
                    'overview': movie.get('overview'),
                    'genres': self._parse_genres(movie.get('genres', [])),
                    'vote_average': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count'),
                    'runtime': movie.get('runtime'),
                    'release_date': movie.get('release_date'),
                    'poster_path': movie.get('poster_path'),
                    'backdrop_path': movie.get('backdrop_path'),
                    'rank': idx + 1
                }
                formatted_results.append(formatted_movie)

            return {
                'movies': formatted_results,
                'total': len(formatted_results),
                'query': query,
                'search_type': 'traditional'
            }

        except Exception as e:
            logger.error(f"❌ Traditional search failed: {e}")
            return {
                'movies': [],
                'total': 0,
                'query': query,
                'search_type': 'traditional',
                'error': str(e)
            }

    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query

        Args:
            query: Partial query string
            limit: Maximum number of suggestions

        Returns:
            List of suggested search terms
        """
        # This could be implemented with your movie data
        # For now, return empty list (frontend handles gracefully)
        return []

    def _parse_genres(self, genres_data) -> List[str]:
        """
        Parse genres from various formats

        Args:
            genres_data: Genres in various formats (string, list, etc.)

        Returns:
            List of genre strings
        """
        if not genres_data:
            return []

        if isinstance(genres_data, str):
            # Handle pipe-separated genres (common in parquet files)
            if '|' in genres_data:
                return [g.strip() for g in genres_data.split('|') if g.strip()]
            # Handle comma-separated genres
            elif ',' in genres_data:
                return [g.strip() for g in genres_data.split(',') if g.strip()]
            else:
                return [genres_data.strip()]

        elif isinstance(genres_data, list):
            return [str(g).strip() for g in genres_data if g]

        else:
            return [str(genres_data).strip()] if genres_data else []

    def is_available(self) -> bool:
        """Check if search service is available"""
        return self.semantic_engine is not None

    def get_status(self) -> Dict[str, Any]:
        """Get search service status information"""
        return {
            'semantic_engine_available': self.semantic_engine is not None,
            'config_path': self.config_path,
            'service': 'SearchService'
        }