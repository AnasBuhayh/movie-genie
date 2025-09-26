"""
Search API Endpoints

Provides semantic and traditional movie search functionality.
"""

from flask import Blueprint, request, current_app
from ..services.search_service import SearchService
from ..utils.responses import APIResponse
from ..utils.errors import ValidationError, ServiceUnavailableError
import logging

logger = logging.getLogger(__name__)

# Create search blueprint
search_bp = Blueprint('search', __name__)

# Initialize service (singleton pattern)
_search_service = None

def get_search_service():
    """Get search service instance"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service

@search_bp.route('/semantic', methods=['GET'])
def semantic_search():
    """
    Semantic movie search endpoint

    Query Parameters:
        q (str): Search query (required)
        k (int): Number of results (optional, default 20)
        user_id (str): User ID for personalization (optional)

    Returns:
        JSON response with search results
    """
    try:
        # Get query parameters
        query = request.args.get('q')
        k = request.args.get('k', 20, type=int)
        user_id = request.args.get('user_id')

        # Validate required parameters
        if not query or query.strip() == '':
            raise ValidationError("Query parameter 'q' is required")

        if k <= 0 or k > 100:
            raise ValidationError("Parameter 'k' must be between 1 and 100")

        logger.info(f"Semantic search request: query='{query}', k={k}, user_id={user_id}")

        # Get search service
        search_service = get_search_service()
        if not search_service.is_available():
            raise ServiceUnavailableError("Semantic Search Engine")

        # Prepare user context for personalization
        user_context = None
        if user_id:
            user_context = {'user_id': user_id}

        # Perform semantic search
        results = search_service.semantic_search(query, k, user_context)

        # Check if search returned error
        if 'error' in results:
            return APIResponse.error(
                message=f"Search failed: {results['error']}",
                error_code="SEARCH_FAILED",
                details=results
            )

        return APIResponse.success(
            data=results,
            message=f"Found {results.get('total', 0)} movies for '{query}'"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except ServiceUnavailableError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return APIResponse.error(
            message="Search request failed",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@search_bp.route('/', methods=['GET'])
@search_bp.route('/traditional', methods=['GET'])
def traditional_search():
    """
    Traditional movie search endpoint (fallback)

    Query Parameters:
        q (str): Search query (required)
        k (int): Number of results (optional, default 20)

    Returns:
        JSON response with search results
    """
    try:
        # Get query parameters
        query = request.args.get('q')
        k = request.args.get('k', 20, type=int)

        # Validate required parameters
        if not query or query.strip() == '':
            raise ValidationError("Query parameter 'q' is required")

        if k <= 0 or k > 100:
            raise ValidationError("Parameter 'k' must be between 1 and 100")

        logger.info(f"Traditional search request: query='{query}', k={k}")

        # Get search service
        search_service = get_search_service()

        # Perform traditional search (falls back to semantic search without personalization)
        results = search_service.traditional_search(query, k)

        return APIResponse.success(
            data=results,
            message=f"Found {results.get('total', 0)} movies for '{query}'"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Traditional search error: {e}")
        return APIResponse.error(
            message="Search request failed",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@search_bp.route('/suggestions', methods=['GET'])
def search_suggestions():
    """
    Search suggestions endpoint

    Query Parameters:
        q (str): Partial search query (required)
        limit (int): Number of suggestions (optional, default 5)

    Returns:
        JSON response with search suggestions
    """
    try:
        # Get query parameters
        query = request.args.get('q', '')
        limit = request.args.get('limit', 5, type=int)

        if len(query) < 2:
            return APIResponse.success(
                data=[],
                message="Query too short for suggestions"
            )

        # Get search service
        search_service = get_search_service()

        # Get suggestions
        suggestions = search_service.get_search_suggestions(query, limit)

        return APIResponse.success(
            data=suggestions,
            message=f"Found {len(suggestions)} suggestions for '{query}'"
        )

    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return APIResponse.error(
            message="Failed to get search suggestions",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@search_bp.route('/status', methods=['GET'])
def search_status():
    """
    Search service status endpoint

    Returns:
        JSON response with service status
    """
    try:
        search_service = get_search_service()
        status = search_service.get_status()

        return APIResponse.success(
            data=status,
            message="Search service status retrieved"
        )

    except Exception as e:
        logger.error(f"Search status error: {e}")
        return APIResponse.error(
            message="Failed to get search status",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )