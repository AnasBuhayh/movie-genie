"""
Feedback API Endpoints

Handles user feedback, ratings, and interaction tracking.
"""

from flask import Blueprint, request, current_app
from datetime import datetime
from ..database.database import DatabaseManager
from ..database.models import User, Interaction
from ..utils.responses import APIResponse
from ..utils.errors import ValidationError, NotFoundError
import logging

logger = logging.getLogger(__name__)

# Create feedback blueprint
feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route('/', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback for a movie

    Request Body:
        {
            "user_id": "optional_user_id",
            "movie_id": 123,
            "rating": 4.5,
            "feedback_type": "like" | "dislike" | "rating" | "watched"
        }

    Returns:
        JSON response with feedback submission status
    """
    try:
        # Get JSON data from request
        data = request.get_json() or {}

        # Extract parameters
        user_id = data.get('user_id')
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        feedback_type = data.get('feedback_type', 'rating')

        # Validate required parameters
        if not movie_id:
            raise ValidationError("Parameter 'movie_id' is required")

        if feedback_type not in ['like', 'dislike', 'rating', 'watched']:
            raise ValidationError("Parameter 'feedback_type' must be one of: like, dislike, rating, watched")

        # Convert feedback to rating scale
        if feedback_type == 'like':
            rating = 5.0
        elif feedback_type == 'dislike':
            rating = 1.0
        elif feedback_type == 'watched':
            rating = rating or 3.0  # Neutral rating if not specified
        elif feedback_type == 'rating':
            if rating is None:
                raise ValidationError("Parameter 'rating' is required for feedback_type 'rating'")
            if not (0.0 <= rating <= 5.0):
                raise ValidationError("Parameter 'rating' must be between 0.0 and 5.0")

        logger.info(f"Feedback submission: user_id={user_id}, movie_id={movie_id}, "
                   f"rating={rating}, type={feedback_type}")

        # Store feedback in database
        feedback_stored = _store_feedback(user_id, movie_id, rating, feedback_type)

        if feedback_stored:
            return APIResponse.success(
                data={
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'feedback_type': feedback_type,
                    'timestamp': datetime.utcnow().isoformat()
                },
                message="Feedback submitted successfully"
            )
        else:
            return APIResponse.error(
                message="Failed to store feedback",
                error_code="STORAGE_ERROR",
                status_code=500
            )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return APIResponse.error(
            message="Failed to submit feedback",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@feedback_bp.route('/rating', methods=['POST'])
def submit_rating():
    """
    Submit a movie rating

    Request Body:
        {
            "user_id": "optional_user_id",
            "movie_id": 123,
            "rating": 4.5
        }

    Returns:
        JSON response with rating submission status
    """
    try:
        # Get JSON data from request
        data = request.get_json() or {}

        # Extract parameters
        user_id = data.get('user_id')
        movie_id = data.get('movie_id')
        rating = data.get('rating')

        # Validate required parameters
        if not movie_id:
            raise ValidationError("Parameter 'movie_id' is required")

        if rating is None:
            raise ValidationError("Parameter 'rating' is required")

        if not (0.0 <= rating <= 5.0):
            raise ValidationError("Parameter 'rating' must be between 0.0 and 5.0")

        logger.info(f"Rating submission: user_id={user_id}, movie_id={movie_id}, rating={rating}")

        # Store rating in database
        rating_stored = _store_feedback(user_id, movie_id, rating, 'rating')

        if rating_stored:
            return APIResponse.success(
                data={
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'feedback_type': 'rating',
                    'timestamp': datetime.utcnow().isoformat()
                },
                message="Rating submitted successfully"
            )
        else:
            return APIResponse.error(
                message="Failed to store rating",
                error_code="STORAGE_ERROR",
                status_code=500
            )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"Rating submission error: {e}")
        return APIResponse.error(
            message="Failed to submit rating",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

@feedback_bp.route('/user/<user_id>/history', methods=['GET'])
def get_user_feedback_history(user_id):
    """
    Get user's feedback history

    Path Parameters:
        user_id (str): User ID

    Query Parameters:
        limit (int): Number of interactions to return (optional, default 50)

    Returns:
        JSON response with user's feedback history
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)

        # Validate parameters
        if limit <= 0 or limit > 1000:
            raise ValidationError("Parameter 'limit' must be between 1 and 1000")

        logger.info(f"User feedback history request: user_id={user_id}, limit={limit}")

        # Get user's interaction history from database
        history = _get_user_history(user_id, limit)

        return APIResponse.success(
            data={
                'user_id': user_id,
                'interactions': history,
                'total': len(history)
            },
            message=f"Retrieved {len(history)} interactions for user {user_id}"
        )

    except ValidationError as e:
        return APIResponse.error(e.message, e.error_code, e.details, e.status_code)
    except Exception as e:
        logger.error(f"User history error: {e}")
        return APIResponse.error(
            message="Failed to retrieve user history",
            error_code="INTERNAL_ERROR",
            details=str(e),
            status_code=500
        )

def _store_feedback(user_id, movie_id, rating, feedback_type):
    """
    Store user feedback in the database

    Args:
        user_id: User ID (optional)
        movie_id: Movie ID
        rating: Rating value
        feedback_type: Type of feedback

    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        # Use the database manager from the app context
        db_manager = current_app.db_manager

        with db_manager.get_session() as session:
            # Create or get user if user_id is provided
            user = None
            if user_id:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    # Create anonymous user entry
                    user = User(
                        id=user_id,
                        email=f"anonymous_{user_id}@moviegenie.local",
                        password_hash="anonymous"
                    )
                    session.add(user)
                    session.flush()  # Get user ID

            # Create interaction record
            interaction = Interaction(
                user_id=user.id if user else None,
                movie_id=movie_id,
                rating=rating,
                interaction_type=feedback_type,
                timestamp=datetime.utcnow()
            )

            session.add(interaction)
            # Session will be committed by context manager

        logger.info(f"Stored feedback: user_id={user_id}, movie_id={movie_id}, "
                   f"rating={rating}, type={feedback_type}")
        return True

    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        return False

def _get_user_history(user_id, limit):
    """
    Get user's interaction history from database

    Args:
        user_id: User ID
        limit: Maximum number of interactions to return

    Returns:
        list: User's interaction history
    """
    try:
        db_manager = current_app.db_manager

        with db_manager.get_session() as session:
            # Get user's interactions
            interactions = (
                session.query(Interaction)
                .filter(Interaction.user_id == user_id)
                .order_by(Interaction.timestamp.desc())
                .limit(limit)
                .all()
            )

            # Format interactions for API response
            history = []
            for interaction in interactions:
                history.append({
                    'movie_id': interaction.movie_id,
                    'rating': interaction.rating,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp.isoformat() if interaction.timestamp else None
                })

            return history

    except Exception as e:
        logger.error(f"Failed to get user history: {e}")
        return []