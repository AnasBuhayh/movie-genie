"""
Users API Endpoints

Provides user information and metadata.
"""

import pandas as pd
from flask import Blueprint, current_app
from ..utils.responses import APIResponse
from ..utils.errors import ServiceUnavailableError
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Create users blueprint
users_bp = Blueprint('users', __name__)

@users_bp.route('/info', methods=['GET'])
def user_info():
    """
    Get user database information

    Returns:
        JSON response with user range and statistics
    """
    try:
        # Get the project root path dynamically
        project_root = Path(__file__).parent.parent.parent.parent.parent
        sequences_path = project_root / 'data' / 'processed' / 'sequences_with_metadata.parquet'

        if not sequences_path.exists():
            raise ServiceUnavailableError("User data not available")

        # Load user sequences to get user info
        logger.info(f"Loading user data from {sequences_path}")
        sequences_df = pd.read_parquet(sequences_path)

        # Get user statistics
        unique_users = sequences_df['userId'].unique()
        min_user_id = int(unique_users.min())
        max_user_id = int(unique_users.max())
        total_users = len(unique_users)

        # Get interaction statistics
        user_stats = sequences_df.groupby('userId').size().describe()

        user_info = {
            'user_id_range': {
                'min': min_user_id,
                'max': max_user_id,
                'total': total_users
            },
            'interaction_stats': {
                'mean_interactions_per_user': round(float(user_stats['mean']), 2),
                'min_interactions': int(user_stats['min']),
                'max_interactions': int(user_stats['max']),
                'median_interactions': int(user_stats['50%'])
            },
            'sample_user_ids': sorted(unique_users[:10].tolist()),
            'instructions': {
                'valid_range': f"Choose a user ID between {min_user_id} and {max_user_id}",
                'note': "Each user has different viewing preferences that affect personalized recommendations"
            }
        }

        logger.info(f"✅ Retrieved user info: {total_users} users, range {min_user_id}-{max_user_id}")

        return APIResponse.success(
            data=user_info,
            message=f"User database contains {total_users} users"
        )

    except Exception as e:
        logger.error(f"User info error: {e}")
        return APIResponse.error(
            message="Failed to retrieve user information",
            error_code="USER_INFO_ERROR",
            details=str(e),
            status_code=500
        )

@users_bp.route('/<int:user_id>/profile', methods=['GET'])
def user_profile(user_id):
    """
    Get user profile information

    Args:
        user_id: User ID to get profile for

    Returns:
        JSON response with user profile and interaction history
    """
    try:
        # Get the project root path dynamically
        project_root = Path(__file__).parent.parent.parent.parent.parent
        sequences_path = project_root / 'data' / 'processed' / 'sequences_with_metadata.parquet'

        if not sequences_path.exists():
            raise ServiceUnavailableError("User data not available")

        # Load user sequences
        logger.info(f"Loading profile for user {user_id}")
        sequences_df = pd.read_parquet(sequences_path)

        # Check if user exists
        user_data = sequences_df[sequences_df['userId'] == user_id]
        if user_data.empty:
            return APIResponse.error(
                message=f"User {user_id} not found",
                error_code="USER_NOT_FOUND",
                status_code=404
            )

        # Get user interaction history
        # Check which columns are available
        available_columns = ['movieId']
        if 'rating' in user_data.columns:
            available_columns.append('rating')
        if 'timestamp' in user_data.columns:
            available_columns.append('timestamp')

        interactions = user_data[available_columns].to_dict('records')

        # Get user statistics
        total_interactions = len(user_data)
        unique_movies = user_data['movieId'].nunique()

        # Get movie titles if available
        try:
            movies_path = project_root / 'data' / 'processed' / 'content_features.parquet'
            if movies_path.exists():
                movies_df = pd.read_parquet(movies_path)
                movie_titles = movies_df.set_index('movieId')['title'].to_dict()

                # Add titles to interactions
                for interaction in interactions:
                    movie_id = interaction['movieId']
                    interaction['title'] = movie_titles.get(movie_id, f"Movie {movie_id}")
        except Exception as e:
            logger.warning(f"Could not load movie titles: {e}")

        user_profile = {
            'user_id': user_id,
            'statistics': {
                'total_interactions': total_interactions,
                'unique_movies_watched': unique_movies,
            },
            'interaction_history': interactions[:20],  # Limit to first 20 for API response
            'total_history_length': len(interactions),
            'recommendation_ready': total_interactions >= 5  # Minimum for good recommendations
        }

        logger.info(f"✅ Retrieved profile for user {user_id}: {total_interactions} interactions")

        return APIResponse.success(
            data=user_profile,
            message=f"User {user_id} profile with {total_interactions} interactions"
        )

    except Exception as e:
        logger.error(f"User profile error: {e}")
        return APIResponse.error(
            message=f"Failed to retrieve user {user_id} profile",
            error_code="USER_PROFILE_ERROR",
            details=str(e),
            status_code=500
        )