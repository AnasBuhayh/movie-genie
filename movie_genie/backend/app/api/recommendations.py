"""
Recommendations API Endpoints

Provides personalized and content-based movie recommendations.
"""

from flask import Blueprint, request, jsonify
from ..services.recommendation_service import RecommendationService
from ..utils.responses import APIResponse
from ..utils.errors import ValidationError, NotFoundError, ServiceUnavailableError
import logging

logger = logging.getLogger(__name__)

# Create recommendations blueprint
recommendations_bp = Blueprint('recommendations', __name__)

# Initialize service (singleton pattern)
_recommendation_service = None

def get_recommendation_service():
    """Get recommendation service instance"""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service

@recommendations_bp.route('/personalized', methods=['POST'])
def get_personalized_recommendations():
    """
    Get personalized movie recommendations

    Request Body:
        {
            "user_id": "optional_user_id",
            "interaction_history": [
                {"movie_id": 123, "rating": 4.5, "timestamp": "2024-01-01T00:00:00Z"},
                ...
            ],
            "limit": 20
        }

    Returns:
        JSON response with personalized recommendations
    """
    try:
        # Get JSON data from request
        data = request.get_json() or {}

        # Extract parameters
        user_id = data.get('user_id')
        interaction_history = data.get('interaction_history', [])
        limit = data.get('limit', 20)

        # Validate parameters
        if limit <= 0 or limit > 100:
            raise ValidationError("Parameter 'limit' must be between 1 and 100")

        logger.info(f"Personalized recommendations request: user_id={user_id}, "
                   f"history_items={len(interaction_history)}, limit={limit}")

        # Get recommendation service
        rec_service = get_recommendation_service()
        if not rec_service.is_available():
            raise ServiceUnavailableError("Recommendation Service")

        # Get personalized recommendations
        recommendations = rec_service.get_personalized_recommendations(
            user_id=user_id,
            interaction_history=interaction_history,
            limit=limit
        )

        return APIResponse.success(
            data=recommendations,
            message=f"Generated {recommendations.get('total', 0)} personalized recommendations"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except ServiceUnavailableError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return APIResponse.error(
            message="Failed to generate personalized recommendations",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@recommendations_bp.route('/', methods=['GET'])
@recommendations_bp.route('/content', methods=['GET'])
def get_content_based_recommendations():
    """
    Get content-based recommendations for a movie

    Query Parameters:
        movie_id (int): Movie ID to base recommendations on (required)
        limit (int): Number of recommendations (optional, default 10)
        type (str): Type of recommendations - 'content_based' or 'similar' (optional)

    Returns:
        JSON response with content-based recommendations
    """
    try:
        # Get query parameters
        movie_id = request.args.get('movie_id', type=int)
        limit = request.args.get('limit', 10, type=int)
        rec_type = request.args.get('type', 'content_based')

        # Validate parameters
        if not movie_id:
            raise ValidationError("Parameter 'movie_id' is required")

        if limit <= 0 or limit > 100:
            raise ValidationError("Parameter 'limit' must be between 1 and 100")

        logger.info(f"Content-based recommendations request: movie_id={movie_id}, "
                   f"limit={limit}, type={rec_type}")

        # Get recommendation service
        rec_service = get_recommendation_service()

        # Get content-based recommendations
        recommendations = rec_service.get_content_based_recommendations(
            movie_id=movie_id,
            limit=limit
        )

        if recommendations.get('total', 0) == 0:
            # Check if source movie exists
            from ..services.movie_service import MovieService
            movie_service = MovieService()
            if not movie_service.get_movie_details(movie_id):
                raise NotFoundError("Movie", movie_id)

        return APIResponse.success(
            data=recommendations,
            message=f"Generated {recommendations.get('total', 0)} content-based recommendations"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except NotFoundError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Content-based recommendations error: {e}")
        return APIResponse.error(
            message="Failed to generate content-based recommendations",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@recommendations_bp.route('/collaborative', methods=['GET'])
def get_collaborative_recommendations():
    """
    Get collaborative filtering recommendations

    Query Parameters:
        user_id (str): User ID for recommendations (required)
        limit (int): Number of recommendations (optional, default 20)

    Returns:
        JSON response with collaborative recommendations
    """
    try:
        # Get query parameters
        user_id = request.args.get('user_id')
        limit = request.args.get('limit', 20, type=int)

        # Validate parameters
        if not user_id:
            raise ValidationError("Parameter 'user_id' is required")

        if limit <= 0 or limit > 100:
            raise ValidationError("Parameter 'limit' must be between 1 and 100")

        logger.info(f"Collaborative recommendations request: user_id={user_id}, limit={limit}")

        # Get recommendation service
        rec_service = get_recommendation_service()
        if not rec_service.is_available():
            raise ServiceUnavailableError("Recommendation Service")

        # Get collaborative recommendations
        recommendations = rec_service.get_collaborative_recommendations(
            user_id=user_id,
            limit=limit
        )

        return APIResponse.success(
            data=recommendations,
            message=f"Generated {recommendations.get('total', 0)} collaborative recommendations"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except ServiceUnavailableError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Collaborative recommendations error: {e}")
        return APIResponse.error(
            message="Failed to generate collaborative recommendations",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@recommendations_bp.route('/popular', methods=['GET'])
def get_popular_recommendations():
    """
    Get popular movie recommendations (fallback)

    Query Parameters:
        limit (int): Number of recommendations (optional, default 20)

    Returns:
        JSON response with popular movie recommendations
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 20, type=int)

        # Validate parameters
        if limit <= 0 or limit > 100:
            raise ValidationError("Parameter 'limit' must be between 1 and 100")

        logger.info(f"Popular recommendations request: limit={limit}")

        # Get recommendation service
        rec_service = get_recommendation_service()

        # Get popular recommendations (fallback)
        recommendations = rec_service._fallback_recommendations(limit)

        return APIResponse.success(
            data=recommendations,
            message=f"Generated {recommendations.get('total', 0)} popular recommendations"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Popular recommendations error: {e}")
        return APIResponse.error(
            message="Failed to generate popular recommendations",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@recommendations_bp.route('/status', methods=['GET'])
def recommendation_service_status():
    """
    Recommendation service status endpoint

    Returns:
        JSON response with service status
    """
    try:
        rec_service = get_recommendation_service()
        status = rec_service.get_status()

        return APIResponse.success(
            data=status,
            message="Recommendation service status retrieved"
        )

    except Exception as e:
        logger.error(f"Recommendation service status error: {e}")
        return APIResponse.error(
            message="Failed to get recommendation service status",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )