"""
Service Layer for Movie Genie Backend

This layer connects your ML infrastructure to the Flask API.
"""

from .search_service import SearchService
from .movie_service import MovieService
from .recommendation_service import RecommendationService

__all__ = ['SearchService', 'MovieService', 'RecommendationService']