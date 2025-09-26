"""
Movie Service - Handles movie data operations

This service provides access to movie information and metadata.
"""

import logging
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class MovieService:
    """Service for handling movie data operations"""

    def __init__(self, content_features_path: Optional[str] = None):
        """
        Initialize movie service

        Args:
            content_features_path: Path to content features parquet file
        """
        self.content_features_path = content_features_path or str(
            project_root / "data" / "processed" / "content_features.parquet"
        )
        self.movies_df = None
        self._load_movie_data()

    def _load_movie_data(self):
        """Load movie data from parquet files"""
        try:
            logger.info(f"Loading movie data from {self.content_features_path}")

            if Path(self.content_features_path).exists():
                self.movies_df = pd.read_parquet(self.content_features_path)
                logger.info(f"✅ Loaded {len(self.movies_df)} movies from content features")
            else:
                logger.error(f"❌ Movie data file not found: {self.content_features_path}")
                self.movies_df = None

        except Exception as e:
            logger.error(f"❌ Failed to load movie data: {e}")
            self.movies_df = None

    def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific movie

        Args:
            movie_id: Movie ID to look up

        Returns:
            Movie details dictionary or None if not found
        """
        if self.movies_df is None:
            logger.error("Movie data not available")
            return None

        try:
            # Look up movie by movieId
            movie_rows = self.movies_df[self.movies_df['movieId'] == movie_id]

            if movie_rows.empty:
                logger.warning(f"Movie {movie_id} not found")
                return None

            movie = movie_rows.iloc[0]

            # Format movie details for API response
            movie_details = {
                'movieId': int(movie.get('movieId', 0)),
                'title': str(movie.get('title', '')),
                'overview': str(movie.get('overview', '')),
                'genres': self._parse_genres(movie.get('genres', [])),
                'release_date': str(movie.get('release_date', '')) if pd.notna(movie.get('release_date')) else None,
                'vote_average': float(movie.get('vote_average', 0)) if pd.notna(movie.get('vote_average')) else None,
                'vote_count': int(movie.get('vote_count', 0)) if pd.notna(movie.get('vote_count')) else None,
                'runtime': int(movie.get('runtime', 0)) if pd.notna(movie.get('runtime')) else None,
                'budget': int(movie.get('budget', 0)) if pd.notna(movie.get('budget')) else None,
                'revenue': int(movie.get('revenue', 0)) if pd.notna(movie.get('revenue')) else None,
                'popularity': float(movie.get('popularity', 0)) if pd.notna(movie.get('popularity')) else None,
                'poster_path': str(movie.get('poster_path', '')) if pd.notna(movie.get('poster_path')) else None,
                'backdrop_path': str(movie.get('backdrop_path', '')) if pd.notna(movie.get('backdrop_path')) else None,
                'imdb_id': str(movie.get('imdb_id', '')) if pd.notna(movie.get('imdb_id')) else None,
                'tmdb_id': int(movie.get('tmdbId', 0)) if pd.notna(movie.get('tmdbId')) else None,
            }

            logger.info(f"✅ Found movie details for {movie_id}: {movie_details['title']}")
            return movie_details

        except Exception as e:
            logger.error(f"❌ Error getting movie details for {movie_id}: {e}")
            return None

    def get_popular_movies(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get popular movies based on vote count and average

        Args:
            limit: Maximum number of movies to return

        Returns:
            List of popular movies
        """
        if self.movies_df is None:
            logger.error("Movie data not available")
            return []

        try:
            # Sort by popularity or vote metrics
            popular_movies = self.movies_df.copy()

            # Filter movies with sufficient vote count
            popular_movies = popular_movies[
                (popular_movies['vote_count'] >= 100) &
                (popular_movies['vote_average'] >= 6.0)
            ]

            # Sort by popularity or vote average
            if 'popularity' in popular_movies.columns:
                popular_movies = popular_movies.sort_values('popularity', ascending=False)
            else:
                popular_movies = popular_movies.sort_values(['vote_average', 'vote_count'], ascending=False)

            # Take top movies
            popular_movies = popular_movies.head(limit)

            # Format for API response
            movies_list = []
            for _, movie in popular_movies.iterrows():
                movie_dict = {
                    'movieId': int(movie.get('movieId', 0)),
                    'title': str(movie.get('title', '')),
                    'overview': str(movie.get('overview', '')),
                    'genres': self._parse_genres(movie.get('genres', [])),
                    'vote_average': float(movie.get('vote_average', 0)) if pd.notna(movie.get('vote_average')) else None,
                    'vote_count': int(movie.get('vote_count', 0)) if pd.notna(movie.get('vote_count')) else None,
                    'popularity': float(movie.get('popularity', 0)) if pd.notna(movie.get('popularity')) else None,
                    'release_date': str(movie.get('release_date', '')) if pd.notna(movie.get('release_date')) else None,
                    'poster_path': str(movie.get('poster_path', '')) if pd.notna(movie.get('poster_path')) else None,
                }
                movies_list.append(movie_dict)

            logger.info(f"✅ Retrieved {len(movies_list)} popular movies")
            return movies_list

        except Exception as e:
            logger.error(f"❌ Error getting popular movies: {e}")
            return []

    def search_movies_by_title(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Simple title-based movie search (fallback for semantic search)

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching movies
        """
        if self.movies_df is None:
            logger.error("Movie data not available")
            return []

        try:
            # Simple case-insensitive title search
            query_lower = query.lower()
            matching_movies = self.movies_df[
                self.movies_df['title'].str.lower().str.contains(query_lower, na=False)
            ]

            # Sort by popularity/vote average
            if 'popularity' in matching_movies.columns:
                matching_movies = matching_movies.sort_values('popularity', ascending=False)
            else:
                matching_movies = matching_movies.sort_values('vote_average', ascending=False)

            matching_movies = matching_movies.head(limit)

            # Format for API response
            movies_list = []
            for _, movie in matching_movies.iterrows():
                movie_dict = {
                    'movieId': int(movie.get('movieId', 0)),
                    'title': str(movie.get('title', '')),
                    'overview': str(movie.get('overview', '')),
                    'genres': self._parse_genres(movie.get('genres', [])),
                    'vote_average': float(movie.get('vote_average', 0)) if pd.notna(movie.get('vote_average')) else None,
                    'release_date': str(movie.get('release_date', '')) if pd.notna(movie.get('release_date')) else None,
                }
                movies_list.append(movie_dict)

            logger.info(f"✅ Found {len(movies_list)} movies matching '{query}'")
            return movies_list

        except Exception as e:
            logger.error(f"❌ Error searching movies by title: {e}")
            return []

    def _parse_genres(self, genres_data) -> List[str]:
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

    def is_available(self) -> bool:
        """Check if movie service is available"""
        return self.movies_df is not None

    def get_status(self) -> Dict[str, Any]:
        """Get movie service status information"""
        return {
            'movies_loaded': self.movies_df is not None,
            'total_movies': len(self.movies_df) if self.movies_df is not None else 0,
            'data_path': self.content_features_path,
            'service': 'MovieService'
        }