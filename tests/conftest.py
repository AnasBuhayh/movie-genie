"""
Pytest configuration and shared fixtures for movie-genie tests.

This file contains shared fixtures and configuration that can be used across
all test files in the movie-genie project, particularly for the two-tower
recommendation model testing.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import warnings

# Set random seeds for reproducible tests
np.random.seed(42)
torch.manual_seed(42)


@pytest.fixture(scope="session")
def sample_rating_distribution():
    """Standard rating distribution for consistent testing."""
    return {
        'positive_ratio': 0.6,  # 60% positive ratings (thumbs up + two thumbs up)
        'negative_ratio': 0.2,  # 20% negative ratings (thumbs down)
        'two_thumbs_ratio': 0.2  # 20% of all ratings are two thumbs up
    }


@pytest.fixture(scope="session")
def standard_movie_features():
    """Standard movie feature configuration for consistent testing."""
    return {
        'numerical_features': ['vote_average', 'vote_count', 'runtime', 'budget', 'revenue'],
        'categorical_features': ['has_budget', 'has_revenue', 'has_runtime', 'is_adult', 'is_independent'],
        'language_features': ['lang_en', 'lang_fr', 'lang_es', 'lang_de', 'lang_ja'],
        'embedding_dim': 768  # Standard EmbeddingGemma dimension
    }


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        yield


def create_realistic_sequences(num_sequences=1000, num_users=100, num_movies=200,
                               start_date=datetime(2018, 1, 1),
                               rating_distribution=None):
    """
    Create realistic sequence data for testing.

    Args:
        num_sequences: Number of rating sequences to generate
        num_users: Number of unique users
        num_movies: Number of unique movies
        start_date: Starting date for temporal sequences
        rating_distribution: Dict with rating distribution parameters

    Returns:
        DataFrame with realistic sequence data
    """
    if rating_distribution is None:
        rating_distribution = {
            'positive_ratio': 0.6,
            'negative_ratio': 0.2,
            'two_thumbs_ratio': 0.2
        }

    sequences = []

    for i in range(num_sequences):
        user_id = (i % num_users) + 1
        movie_id = (i % num_movies) + 1

        # Create temporal progression (sequences get later over time)
        days_offset = i // 10  # Every 10 sequences advance by 1 day
        timestamp = start_date + timedelta(days=days_offset)

        # Generate rating based on user/movie biases and random factors
        user_bias = np.sin(user_id * 0.1)  # User preference pattern
        movie_bias = np.cos(movie_id * 0.05)  # Movie quality pattern
        noise = np.random.normal(0, 0.3)

        combined_score = user_bias + movie_bias + noise

        # Convert to thumbs rating based on distribution
        rand_val = np.random.random()
        if rand_val < rating_distribution['negative_ratio']:
            thumbs_rating = -1.0
        elif rand_val < rating_distribution['negative_ratio'] + rating_distribution['positive_ratio']:
            if np.random.random() < rating_distribution['two_thumbs_ratio'] / rating_distribution['positive_ratio']:
                thumbs_rating = 2.0  # Two thumbs up
            else:
                thumbs_rating = 1.0  # Thumbs up
        else:
            # Use combined score to determine rating for remaining cases
            thumbs_rating = 1.0 if combined_score > 0 else -1.0

        sequences.append({
            'userId': user_id,
            'movieId': movie_id,
            'rating': 3.0 + thumbs_rating,  # Original 1-5 scale rating
            'thumbs_rating': thumbs_rating,
            'datetime': timestamp,
            'sequence_id': f"{user_id}_seq_{i//20}",  # Group into sequences
            'user_category': np.random.choice(['light', 'medium', 'heavy'], p=[0.3, 0.5, 0.2])
        })

    return pd.DataFrame(sequences)


def create_realistic_movies(num_movies=200, standard_features=None):
    """
    Create realistic movie feature data for testing.

    Args:
        num_movies: Number of movies to generate
        standard_features: Dict with feature configuration

    Returns:
        DataFrame with realistic movie features
    """
    if standard_features is None:
        standard_features = {
            'numerical_features': ['vote_average', 'vote_count', 'runtime', 'budget', 'revenue'],
            'categorical_features': ['has_budget', 'has_revenue', 'has_runtime', 'is_adult', 'is_independent'],
            'language_features': ['lang_en', 'lang_fr', 'lang_es'],
            'embedding_dim': 768
        }

    movies = []

    for movie_id in range(1, num_movies + 1):
        # Generate realistic numerical features
        movie_data = {
            'movieId': movie_id,
            'vote_average': np.clip(np.random.normal(6.5, 1.5), 1, 10),
            'vote_count': int(np.random.lognormal(5, 1.5)),
            'runtime': int(np.clip(np.random.normal(110, 25), 60, 200)),
            'budget': int(np.random.lognormal(15, 2)) if np.random.random() > 0.3 else 0,
            'revenue': int(np.random.lognormal(16, 2)) if np.random.random() > 0.2 else 0,
        }

        # Add categorical features
        movie_data.update({
            'has_budget': 1 if movie_data['budget'] > 0 else 0,
            'has_revenue': 1 if movie_data['revenue'] > 0 else 0,
            'has_runtime': 1,
            'is_adult': np.random.choice([0, 1], p=[0.95, 0.05]),
            'is_independent': np.random.choice([0, 1], p=[0.8, 0.2]),
        })

        # Add language features (one-hot encoded)
        languages = ['en', 'fr', 'es', 'de', 'ja', 'zh', 'it', 'ru']
        primary_lang = np.random.choice(languages, p=[0.6, 0.1, 0.08, 0.05, 0.05, 0.04, 0.04, 0.04])

        for lang in languages:
            movie_data[f'lang_{lang}'] = 1 if lang == primary_lang else 0

        # Generate text embedding (simulate EmbeddingGemma output)
        # Quality movies tend to have different embedding patterns
        quality_factor = (movie_data['vote_average'] - 5) / 5  # -1 to 1
        base_embedding = np.random.normal(0, 0.1, standard_features['embedding_dim'])
        quality_signal = np.random.normal(quality_factor * 0.05, 0.02, standard_features['embedding_dim'])

        movie_data['text_embedding'] = (base_embedding + quality_signal).tolist()

        movies.append(movie_data)

    return pd.DataFrame(movies)


@pytest.fixture
def realistic_test_data(sample_rating_distribution, standard_movie_features):
    """
    Create realistic test data for comprehensive testing.

    Returns temporary files with realistic sequences and movie features.
    """
    temp_dir = tempfile.mkdtemp()

    # Generate realistic data
    sequences_df = create_realistic_sequences(
        num_sequences=500,
        num_users=50,
        num_movies=100,
        rating_distribution=sample_rating_distribution
    )

    movies_df = create_realistic_movies(
        num_movies=100,
        standard_features=standard_movie_features
    )

    # Save to temporary files
    sequences_path = Path(temp_dir) / "sequences.parquet"
    movies_path = Path(temp_dir) / "movies.parquet"

    sequences_df.to_parquet(sequences_path)
    movies_df.to_parquet(movies_path)

    yield {
        'sequences_path': str(sequences_path),
        'movies_path': str(movies_path),
        'temp_dir': temp_dir,
        'sequences_df': sequences_df,
        'movies_df': movies_df
    }

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_test_data():
    """Create minimal test data for fast unit tests."""
    temp_dir = tempfile.mkdtemp()

    # Very small dataset for quick tests
    sequences_df = create_realistic_sequences(
        num_sequences=50,
        num_users=5,
        num_movies=10
    )

    movies_df = create_realistic_movies(num_movies=10)

    sequences_path = Path(temp_dir) / "sequences.parquet"
    movies_path = Path(temp_dir) / "movies.parquet"

    sequences_df.to_parquet(sequences_path)
    movies_df.to_parquet(movies_path)

    yield {
        'sequences_path': str(sequences_path),
        'movies_path': str(movies_path),
        'temp_dir': temp_dir
    }

    shutil.rmtree(temp_dir)


# Custom pytest markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "temporal: marks tests related to temporal splitting"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests for edge cases and error handling"
    )


# Pytest collection modifications for better test organization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "integration" in item.nodeid or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark temporal tests
        if "temporal" in item.nodeid or "temporal" in item.name.lower():
            item.add_marker(pytest.mark.temporal)

        # Mark edge case tests
        if "edge" in item.nodeid or "error" in item.name.lower():
            item.add_marker(pytest.mark.edge_case)