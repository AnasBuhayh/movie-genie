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
                    'genres': movie.get('genres', []),
                    'similarity_score': movie.get('similarity_score'),
                    'personalized_score': movie.get('personalized_score'),
                    'rank': movie.get('rank'),
                    'vote_average': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count'),
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
        Perform traditional text-based search (fallback)

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            Dictionary with search results and metadata
        """
        # This is a simplified fallback - you could implement a traditional search here
        # For now, we'll just return semantic search without personalization
        logger.info(f"Performing traditional search for: '{query}' (k={k})")

        return self.semantic_search(query, k, user_context=None)

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