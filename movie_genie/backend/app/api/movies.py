"""
Movie API Endpoints

Provides movie details, popular movies, and movie-related operations.
"""

from flask import Blueprint, request
from ..services.movie_service import MovieService
from ..utils.responses import APIResponse
from ..utils.errors import ValidationError, NotFoundError, ServiceUnavailableError
import logging

logger = logging.getLogger(__name__)

# Create movies blueprint
movies_bp = Blueprint('movies', __name__)

# Initialize service (singleton pattern)
_movie_service = None

def get_movie_service():
    """Get movie service instance"""
    global _movie_service
    if _movie_service is None:
        _movie_service = MovieService()
    return _movie_service

@movies_bp.route('/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """
    Get detailed information for a specific movie

    Path Parameters:
        movie_id (int): Movie ID

    Returns:
        JSON response with movie details
    """
    try:
        logger.info(f"Movie details request: movie_id={movie_id}")

        # Get movie service
        movie_service = get_movie_service()
        if not movie_service.is_available():
            raise ServiceUnavailableError("Movie Service")

        # Get movie details
        movie_details = movie_service.get_movie_details(movie_id)

        if not movie_details:
            raise NotFoundError("Movie", movie_id)

        return APIResponse.success(
            data=movie_details,
            message=f"Movie details retrieved for ID {movie_id}"
        )

    except NotFoundError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except ServiceUnavailableError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Movie details error: {e}")
        return APIResponse.error(
            message="Failed to retrieve movie details",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@movies_bp.route('/popular', methods=['GET'])
def get_popular_movies():
    """
    Get popular movies

    Query Parameters:
        limit (int): Number of movies to return (optional, default 20, max 100)

    Returns:
        JSON response with popular movies
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 20, type=int)

        # Validate parameters
        if limit <= 0 or limit > 100:
            raise ValidationError("Parameter 'limit' must be between 1 and 100")

        logger.info(f"Popular movies request: limit={limit}")

        # Get movie service
        movie_service = get_movie_service()
        if not movie_service.is_available():
            raise ServiceUnavailableError("Movie Service")

        # Get popular movies
        popular_movies = movie_service.get_popular_movies(limit)

        # Format response to match frontend expectations
        response_data = {
            'movies': popular_movies,
            'recommendation_type': 'popular',
            'total': len(popular_movies)
        }

        return APIResponse.success(
            data=response_data,
            message=f"Retrieved {len(popular_movies)} popular movies"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except ServiceUnavailableError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Popular movies error: {e}")
        return APIResponse.error(
            message="Failed to retrieve popular movies",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@movies_bp.route('/search', methods=['GET'])
def search_movies_by_title():
    """
    Simple title-based movie search (fallback)

    Query Parameters:
        q (str): Search query (required)
        limit (int): Number of results (optional, default 20)

    Returns:
        JSON response with search results
    """
    try:
        # Get query parameters
        query = request.args.get('q')
        limit = request.args.get('limit', 20, type=int)

        # Validate parameters
        if not query or query.strip() == '':
            raise ValidationError("Query parameter 'q' is required")

        if limit <= 0 or limit > 100:
            raise ValidationError("Parameter 'limit' must be between 1 and 100")

        logger.info(f"Movie title search request: query='{query}', limit={limit}")

        # Get movie service
        movie_service = get_movie_service()
        if not movie_service.is_available():
            raise ServiceUnavailableError("Movie Service")

        # Search movies by title
        movies = movie_service.search_movies_by_title(query, limit)

        # Format response to match search API format
        response_data = {
            'movies': movies,
            'total': len(movies),
            'query': query,
            'search_type': 'title_search'
        }

        return APIResponse.success(
            data=response_data,
            message=f"Found {len(movies)} movies matching '{query}'"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except ServiceUnavailableError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Movie title search error: {e}")
        return APIResponse.error(
            message="Movie search failed",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@movies_bp.route('/genres', methods=['GET'])
def get_movie_genres():
    """
    Get available movie genres

    Returns:
        JSON response with list of genres
    """
    try:
        logger.info("Movie genres request")

        # Get movie service
        movie_service = get_movie_service()
        if not movie_service.is_available():
            raise ServiceUnavailableError("Movie Service")

        # This is a basic implementation - you could enhance this
        # by extracting unique genres from your movie dataset
        common_genres = [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
            "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
            "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
        ]

        return APIResponse.success(
            data=common_genres,
            message=f"Retrieved {len(common_genres)} movie genres"
        )

    except ServiceUnavailableError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Movie genres error: {e}")
        return APIResponse.error(
            message="Failed to retrieve movie genres",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@movies_bp.route('/status', methods=['GET'])
def movie_service_status():
    """
    Movie service status endpoint

    Returns:
        JSON response with service status
    """
    try:
        movie_service = get_movie_service()
        status = movie_service.get_status()

        return APIResponse.success(
            data=status,
            message="Movie service status retrieved"
        )

    except Exception as e:
        logger.error(f"Movie service status error: {e}")
        return APIResponse.error(
            message="Failed to get movie service status",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )