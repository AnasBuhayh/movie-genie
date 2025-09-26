"""
Feature Processing and Engineering

This module contains feature processors for different data sources
in the Movie Genie recommendation system.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .embeddings import TextEmbedder, preprocess_movie_text, preprocess_review_text


class TMDBFeatureProcessor:
    """Processes TMDB movie data to extract and engineer features."""

    # Constants
    TOP_LANGUAGES_COUNT = 15
    SEASON_MAPPING = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11]
    }

    def __init__(self, embedder: Optional[TextEmbedder] = None):
        """Initialize the TMDB feature processor.

        Args:
            embedder: Text embedder for generating content embeddings
        """
        self.embedder = embedder or TextEmbedder()

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps to TMDB data.

        Args:
            df: DataFrame with raw TMDB movie data

        Returns:
            DataFrame with engineered features
        """
        logging.info("Starting TMDB feature processing...")

        df = self._extract_numerical_features(df)
        df = self._extract_categorical_features(df)
        df = self._extract_language_features(df)
        df = self._extract_date_features(df)
        df = self._extract_production_features(df)
        df = self._extract_text_features(df)

        logging.info("TMDB feature processing completed")
        return df

    def _extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and clean numerical features."""
        logging.info("Extracting numerical features...")

        # Budget features
        df['budget'] = df['budget'].fillna(0)
        df['has_budget'] = (df['budget'] > 0).astype(int)

        # Revenue features
        df['revenue'] = df['revenue'].fillna(0)
        df['has_revenue'] = (df['revenue'] > 0).astype(int)

        # Runtime features
        df['runtime'] = df['runtime'].fillna(0)
        df['has_runtime'] = (df['runtime'] > 0).astype(int)

        # Rating features
        df['vote_average'] = df['vote_average'].fillna(0)
        df['vote_count'] = df['vote_count'].fillna(0)

        # Popularity score
        df['popularity'] = df['popularity'].fillna(0)

        # Derived features
        df['roi'] = np.where(
            df['budget'] > 0,
            df['revenue'] / df['budget'],
            0
        )

        return df

    def _extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode categorical features."""
        logging.info("Processing categorical features...")

        # Adult content flag
        df['is_adult'] = df['adult'].fillna(False).astype(int)

        return df

    def _extract_language_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode language features."""
        # Handle missing language data
        df['original_language'] = df['original_language'].fillna('unknown')

        # Get language distribution
        lang_counts = df['original_language'].value_counts()
        top_languages = lang_counts.head(self.TOP_LANGUAGES_COUNT).index.tolist()

        logging.info(f"Encoding top {self.TOP_LANGUAGES_COUNT} languages: {top_languages}")

        # One-hot encode top languages
        for lang in top_languages:
            col_name = f'lang_{lang}'
            df[col_name] = (df['original_language'] == lang).astype(int)

        # Create 'other' category
        df['lang_other'] = (~df['original_language'].isin(top_languages)).astype(int)

        return df

    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract date and time-based features."""
        # Parse release dates
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_decade'] = (df['release_year'] // 10) * 10
        df['release_month'] = df['release_date'].dt.month

        # Add season feature
        df['release_season'] = df['release_month'].apply(self._get_season_from_month)

        return df

    def _get_season_from_month(self, month: Optional[int]) -> str:
        """Convert month number to season."""
        if pd.isna(month):
            return 'unknown'

        for season, months in self.SEASON_MAPPING.items():
            if month in months:
                return season

        return 'unknown'

    def _extract_production_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract production company features."""
        # Parse production companies
        df['company_list'] = df['production_companies'].apply(self._parse_production_companies)

        # Extract primary production company
        df['primary_company'] = df['company_list'].apply(
            lambda companies: companies[0] if companies else 'unknown'
        )

        # Count of production companies
        df['num_companies'] = df['company_list'].apply(len)

        # Independent vs studio system
        df['is_independent'] = (df['num_companies'] <= 1).astype(int)

        return df

    def _parse_production_companies(self, company_string: str) -> List[str]:
        """Parse production companies JSON string."""
        if pd.isna(company_string) or company_string == "[]":
            return []

        try:
            companies_list = json.loads(company_string.replace("'", '"'))
            return [company['name'] for company in companies_list]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text features and embeddings."""
        logging.info("Processing text features...")

        # Combine text fields
        df['combined_text'] = df.apply(
            lambda row: preprocess_movie_text(
                overview=row.get('overview'),
                tagline=row.get('tagline'),
                keywords=self._parse_keywords(row.get('keywords'))
            ), axis=1
        )

        # Generate embeddings
        if self.embedder.is_available():
            logging.info("Generating text embeddings...")
            embeddings = self.embedder.embed_texts(df['combined_text'].tolist())
            df['text_embedding'] = embeddings.tolist()
        else:
            logging.info("No model available, skipping embedding generation")
            df['text_embedding'] = None

        return df

    def _parse_keywords(self, keyword_string: str) -> List[str]:
        """Parse keywords JSON string."""
        if pd.isna(keyword_string) or keyword_string == "[]":
            return []

        try:
            keywords_list = json.loads(keyword_string.replace("'", '"'))
            return [keyword['name'] for keyword in keywords_list]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []


class IMDbReviewProcessor:
    """Processes IMDb review data to extract features and embeddings."""

    def __init__(self, embedder: Optional[TextEmbedder] = None):
        """Initialize the IMDb review processor.

        Args:
            embedder: Text embedder for generating review embeddings
        """
        self.embedder = embedder or TextEmbedder()

    def process_reviews(self, reviews_df: pd.DataFrame, links_df: pd.DataFrame) -> pd.DataFrame:
        """Process IMDb reviews and merge with MovieLens data.

        Args:
            reviews_df: DataFrame with IMDb review data
            links_df: DataFrame with MovieLens links data

        Returns:
            DataFrame with processed reviews and MovieLens IDs
        """
        logging.info("Starting IMDb review processing...")

        # Convert IMDb ID format and merge with links
        processed_df = self._merge_with_movielens(reviews_df, links_df)

        # Extract text features and embeddings
        processed_df = self._extract_text_features(processed_df)

        logging.info(f"Processed {len(processed_df):,} reviews for {processed_df['movieId'].nunique():,} movies")
        return processed_df

    def _merge_with_movielens(self, reviews_df: pd.DataFrame, links_df: pd.DataFrame) -> pd.DataFrame:
        """Merge IMDb reviews with MovieLens links."""
        # Convert IMDb ID format: tt0123456 -> 123456
        reviews_df['imdb_id_int'] = reviews_df['imdb_id'].str.replace('tt', '').astype(int)

        # Merge with links to get MovieLens IDs
        merged_df = pd.merge(
            reviews_df,
            links_df,
            left_on='imdb_id_int',
            right_on='imdbId',
            how='inner'
        )

        logging.info(f"Merged {len(merged_df):,} reviews with MovieLens data")
        return merged_df

    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text features and embeddings from reviews."""
        logging.info("Processing review text features...")

        # Combine title and content
        df['combined_text'] = df.apply(
            lambda row: preprocess_review_text(
                title=row.get('title'),
                content=row.get('content')
            ), axis=1
        )

        # Generate embeddings
        if self.embedder.is_available():
            logging.info("Generating review embeddings...")
            embeddings = self.embedder.embed_texts(df['combined_text'].tolist())
            df['review_embedding'] = embeddings.tolist()
        else:
            logging.info("No model available, skipping embedding generation")
            df['review_embedding'] = None

        return df


class DataMerger:
    """Handles merging of different data sources."""

    @staticmethod
    def merge_tmdb_with_links(links_df: pd.DataFrame, tmdb_df: pd.DataFrame) -> pd.DataFrame:
        """Merge MovieLens links with TMDB data.

        Args:
            links_df: MovieLens links DataFrame
            tmdb_df: TMDB movies DataFrame

        Returns:
            Merged DataFrame
        """
        enriched_movies = links_df.merge(
            tmdb_df,
            left_on='tmdbId',
            right_on='id',
            how='left'
        )

        logging.info(f"Merged datasets: {len(enriched_movies):,} movies")

        # Check TMDB coverage
        has_tmdb_data = enriched_movies['title'].notna()
        coverage_pct = has_tmdb_data.mean() * 100
        logging.info(f"TMDB coverage: {has_tmdb_data.sum():,} movies ({coverage_pct:.1f}%)")

        return enriched_movies

    @staticmethod
    def filter_by_ratings(links_df: pd.DataFrame, ratings_path: Path) -> pd.DataFrame:
        """Filter links to only include movies that have ratings.

        Args:
            links_df: MovieLens links DataFrame
            ratings_path: Path to ratings.csv file

        Returns:
            Filtered links DataFrame
        """
        if not ratings_path.exists():
            logging.warning(f"Ratings file not found: {ratings_path}")
            return links_df

        # Load ratings and get unique movie IDs
        ratings_df = pd.read_csv(ratings_path)
        movies_with_ratings = set(ratings_df['movieId'].unique())

        # Filter links
        filtered_links = links_df[links_df['movieId'].isin(movies_with_ratings)]

        logging.info(f"Filtered links: {len(filtered_links):,} movies with ratings "
                    f"(from {len(links_df):,} total)")

        return filtered_links