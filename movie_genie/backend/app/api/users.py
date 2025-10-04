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

@users_bp.route('/<int:user_id>/watched', methods=['GET'])
def user_watched_movies(user_id):
    """
    Get movies watched/rated by a specific user

    Args:
        user_id: User ID to get watched movies for

    Query Parameters:
        limit (int): Maximum number of movies to return (default: 20, max: 100)

    Returns:
        JSON response with user's watched movies
    """
    try:
        from flask import request

        # Get query parameters
        limit = request.args.get('limit', 20, type=int)

        # Validate limit
        if limit <= 0 or limit > 100:
            return APIResponse.error(
                message="Parameter 'limit' must be between 1 and 100",
                error_code="VALIDATION_ERROR",
                status_code=400
            )

        # Get the project root path dynamically
        project_root = Path(__file__).parent.parent.parent.parent.parent
        sequences_path = project_root / 'data' / 'processed' / 'sequences_with_metadata.parquet'
        movies_path = project_root / 'data' / 'processed' / 'content_features.parquet'

        if not sequences_path.exists():
            raise ServiceUnavailableError("User data not available")

        if not movies_path.exists():
            raise ServiceUnavailableError("Movie data not available")

        logger.info(f"Loading watched movies for user {user_id}")

        # Load user sequences
        sequences_df = pd.read_parquet(sequences_path)

        # Check if user exists
        user_data = sequences_df[sequences_df['userId'] == user_id]
        if user_data.empty:
            return APIResponse.error(
                message=f"User {user_id} not found",
                error_code="USER_NOT_FOUND",
                status_code=404
            )

        # Get movie IDs the user has interacted with
        # Sort by timestamp if available to get most recent first
        if 'timestamp' in user_data.columns:
            user_data = user_data.sort_values('timestamp', ascending=False)

        watched_movie_ids = user_data['movieId'].head(limit).unique()

        # Load movie details
        movies_df = pd.read_parquet(movies_path)
        watched_movies_data = movies_df[movies_df['movieId'].isin(watched_movie_ids)]

        # Format movies for API response
        watched_movies = []
        for _, movie in watched_movies_data.iterrows():
            movie_dict = {
                'movieId': int(movie.get('movieId', 0)),
                'title': str(movie.get('title', '')),
                'overview': str(movie.get('overview', '')) if pd.notna(movie.get('overview')) else '',
                'genres': _parse_genres(movie.get('genres', [])),
                'vote_average': float(movie.get('vote_average', 0)) if pd.notna(movie.get('vote_average')) else None,
                'vote_count': int(movie.get('vote_count', 0)) if pd.notna(movie.get('vote_count')) else None,
                'release_date': str(movie.get('release_date', '')) if pd.notna(movie.get('release_date')) else None,
                'poster_path': str(movie.get('poster_path', '')) if pd.notna(movie.get('poster_path')) else None,
                'watched': True  # Mark as watched
            }
            watched_movies.append(movie_dict)

        response_data = {
            'movies': watched_movies,
            'recommendation_type': 'watched',
            'total': len(watched_movies),
            'user_id': user_id
        }

        logger.info(f"✅ Retrieved {len(watched_movies)} watched movies for user {user_id}")

        return APIResponse.success(
            data=response_data,
            message=f"Retrieved {len(watched_movies)} watched movies for user {user_id}"
        )

    except Exception as e:
        logger.error(f"User watched movies error: {e}")
        return APIResponse.error(
            message=f"Failed to retrieve watched movies for user {user_id}",
            error_code="USER_WATCHED_ERROR",
            details=str(e),
            status_code=500
        )

@users_bp.route('/<int:user_id>/historical-interest', methods=['GET'])
def user_historical_interest(user_id):
    """
    Get movies based on user's historical genre/temporal interests

    Analyzes user's past viewing patterns to find movies from their
    preferred genres and time periods.

    Args:
        user_id: User ID to get historical interest for

    Query Parameters:
        limit (int): Maximum number of movies to return (default: 20, max: 100)

    Returns:
        JSON response with movies matching user's historical interests
    """
    try:
        from flask import request
        from collections import Counter

        # Get query parameters
        limit = request.args.get('limit', 20, type=int)

        # Validate limit
        if limit <= 0 or limit > 100:
            return APIResponse.error(
                message="Parameter 'limit' must be between 1 and 100",
                error_code="VALIDATION_ERROR",
                status_code=400
            )

        # Get the project root path dynamically
        project_root = Path(__file__).parent.parent.parent.parent.parent
        sequences_path = project_root / 'data' / 'processed' / 'sequences_with_metadata.parquet'
        movies_path = project_root / 'data' / 'processed' / 'content_features.parquet'

        if not sequences_path.exists():
            raise ServiceUnavailableError("User data not available")

        if not movies_path.exists():
            raise ServiceUnavailableError("Movie data not available")

        logger.info(f"Analyzing historical interest for user {user_id}")

        # Load data
        sequences_df = pd.read_parquet(sequences_path)
        movies_df = pd.read_parquet(movies_path)

        # Check if user exists
        user_data = sequences_df[sequences_df['userId'] == user_id]
        if user_data.empty:
            return APIResponse.error(
                message=f"User {user_id} not found",
                error_code="USER_NOT_FOUND",
                status_code=404
            )

        # Get user's watched movies
        watched_movie_ids = user_data['movieId'].unique()
        watched_movies = movies_df[movies_df['movieId'].isin(watched_movie_ids)]

        # Analyze user's genre preferences
        genre_counts = Counter()
        for genres in watched_movies['genres'].dropna():
            if isinstance(genres, str):
                genre_list = [g.strip() for g in genres.split(',')]
            elif isinstance(genres, list):
                genre_list = genres
            else:
                continue

            for genre in genre_list:
                if genre:
                    genre_counts[genre] += 1

        # Get top 3 favorite genres
        favorite_genres = [genre for genre, _ in genre_counts.most_common(3)]

        if not favorite_genres:
            # Fallback to popular movies if no genre preferences found
            logger.warning(f"No genre preferences found for user {user_id}, using popular movies")
            popular_movies = movies_df[
                (movies_df['vote_count'] >= 50) &
                (movies_df['vote_average'] >= 6.0)
            ].sort_values('popularity', ascending=False).head(limit)
        else:
            logger.info(f"User {user_id} favorite genres: {favorite_genres}")

            # Find movies from favorite genres that user hasn't watched
            def has_favorite_genre(genres_data):
                if pd.isna(genres_data) or not genres_data:
                    return False

                if isinstance(genres_data, str):
                    genres = [g.strip() for g in genres_data.split(',')]
                elif isinstance(genres_data, list):
                    genres = genres_data
                else:
                    return False

                return any(fav_genre in genres for fav_genre in favorite_genres)

            # Filter movies by favorite genres
            genre_matches = movies_df[
                movies_df['genres'].apply(has_favorite_genre) &
                ~movies_df['movieId'].isin(watched_movie_ids)  # Exclude already watched
            ]

            # Sort by popularity and rating
            if 'popularity' in genre_matches.columns:
                popular_movies = genre_matches[
                    (genre_matches['vote_count'] >= 20) &
                    (genre_matches['vote_average'] >= 6.0)
                ].sort_values('popularity', ascending=False).head(limit)
            else:
                popular_movies = genre_matches[
                    (genre_matches['vote_count'] >= 20) &
                    (genre_matches['vote_average'] >= 6.0)
                ].sort_values(['vote_average', 'vote_count'], ascending=False).head(limit)

        # Format movies for API response
        historical_movies = []
        for _, movie in popular_movies.iterrows():
            movie_dict = {
                'movieId': int(movie.get('movieId', 0)),
                'title': str(movie.get('title', '')),
                'overview': str(movie.get('overview', '')) if pd.notna(movie.get('overview')) else '',
                'genres': _parse_genres(movie.get('genres', [])),
                'vote_average': float(movie.get('vote_average', 0)) if pd.notna(movie.get('vote_average')) else None,
                'vote_count': int(movie.get('vote_count', 0)) if pd.notna(movie.get('vote_count')) else None,
                'popularity': float(movie.get('popularity', 0)) if pd.notna(movie.get('popularity')) else None,
                'release_date': str(movie.get('release_date', '')) if pd.notna(movie.get('release_date')) else None,
                'poster_path': str(movie.get('poster_path', '')) if pd.notna(movie.get('poster_path')) else None,
            }
            historical_movies.append(movie_dict)

        response_data = {
            'movies': historical_movies,
            'recommendation_type': 'historical_interest',
            'total': len(historical_movies),
            'user_id': user_id,
            'favorite_genres': favorite_genres,
            'analysis': {
                'watched_movies_count': len(watched_movie_ids),
                'top_genres': favorite_genres
            }
        }

        logger.info(f"✅ Retrieved {len(historical_movies)} historical interest movies for user {user_id}")

        return APIResponse.success(
            data=response_data,
            message=f"Retrieved {len(historical_movies)} movies based on historical interest"
        )

    except Exception as e:
        logger.error(f"Historical interest error: {e}")
        return APIResponse.error(
            message=f"Failed to retrieve historical interest for user {user_id}",
            error_code="HISTORICAL_INTEREST_ERROR",
            details=str(e),
            status_code=500
        )

def _parse_genres(genres_data):
    """
    Parse genres from various formats

    Args:
        genres_data: Genres in various formats (string, list, etc.)

    Returns:
        List of genre strings
    """
    if pd.isna(genres_data) or not genres_data:
        return []

    if isinstance(genres_data, str):
        # Handle comma-separated genres or JSON-like strings
        if ',' in genres_data:
            return [g.strip() for g in genres_data.split(',')]
        else:
            return [genres_data.strip()]

    elif isinstance(genres_data, list):
        return [str(g).strip() for g in genres_data]

    else:
        return [str(genres_data).strip()]