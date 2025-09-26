import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import warnings

from movie_genie.retrieval.two_tower_model import (
    TwoTowerDataLoader, TwoTowerTrainer, TwoTowerModel,
    TwoTowerEvaluator, TwoTowerDataset
)


class TestEdgeCases:
    """Test edge cases and error handling in the two-tower model."""

    @pytest.fixture
    def minimal_valid_data(self):
        """Create minimal valid data for edge case testing."""
        temp_dir = tempfile.mkdtemp()

        # Minimal sequences - just enough to pass validation
        sequences = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2],
            'movieId': [1, 2, 3, 1, 2, 4],
            'rating': [4.0, 5.0, 2.0, 3.0, 4.5, 1.0],
            'thumbs_rating': [1.0, 2.0, -1.0, 1.0, 2.0, -1.0],
            'datetime': pd.date_range('2020-01-01', periods=6, freq='D'),
            'sequence_id': ['1_seq_1'] * 3 + ['2_seq_1'] * 3,
            'user_category': ['medium'] * 6
        })

        # Minimal movies - matching the movieIds
        movies = pd.DataFrame({
            'movieId': [1, 2, 3, 4],
            'vote_average': [7.0, 8.0, 5.0, 6.0],
            'vote_count': [100, 200, 50, 150],
            'runtime': [120, 90, 150, 110],
            'budget': [1000000, 2000000, 500000, 1500000],
            'revenue': [5000000, 10000000, 1000000, 7000000],
            'has_budget': [1, 1, 1, 1],
            'has_revenue': [1, 1, 1, 1],
            'has_runtime': [1, 1, 1, 1],
            'is_adult': [0, 0, 0, 0],
            'lang_en': [1, 1, 0, 1],
            'lang_fr': [0, 0, 1, 0],
            'is_independent': [0, 1, 0, 0],
            'text_embedding': [
                np.random.normal(0, 0.1, 768).tolist() for _ in range(4)
            ]
        })

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"

        sequences.to_parquet(sequences_path)
        movies.to_parquet(movies_path)

        yield {
            'sequences_path': str(sequences_path),
            'movies_path': str(movies_path),
            'temp_dir': temp_dir
        }

        shutil.rmtree(temp_dir)

    def test_minimum_viable_training(self, minimal_valid_data):
        """Test training with absolute minimum viable data."""

        data_loader = TwoTowerDataLoader(
            sequences_path=minimal_valid_data['sequences_path'],
            movies_path=minimal_valid_data['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=1  # Very low threshold
        )

        # Should successfully load minimal data
        assert data_loader.num_users >= 1
        assert data_loader.num_movies >= 1
        assert len(data_loader.positive_examples) >= 1

        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=4,  # Very small
            user_hidden_dims=[4],
            item_hidden_dims=[4],
            dropout_rate=0.0
        )

        trainer = TwoTowerTrainer(model=model, data_loader=data_loader)

        # Should complete training even with minimal data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history = trainer.train(num_epochs=1, batch_size=2, validation_split=0.3)

        # Check that training completed successfully (may have different history format)
        assert 'epoch' in history
        assert len(history['epoch']) == 1  # Trained for 1 epoch

    def test_all_positive_ratings(self):
        """Test handling when all ratings are positive."""
        temp_dir = tempfile.mkdtemp()

        # All positive ratings
        sequences = pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [1, 2, 3, 4, 5, 6],
            'rating': [4.0, 5.0, 4.5, 5.0, 4.0, 4.5],
            'thumbs_rating': [1.0, 2.0, 1.0, 2.0, 1.0, 1.0],  # All positive
            'datetime': pd.date_range('2020-01-01', periods=6, freq='D'),
            'sequence_id': ['1_seq_1'] * 2 + ['2_seq_1'] * 2 + ['3_seq_1'] * 2,
            'user_category': ['medium'] * 6
        })

        movies = pd.DataFrame({
            'movieId': list(range(1, 7)),
            'vote_average': [7.0] * 6,
            'vote_count': [100] * 6,
            'runtime': [120] * 6,
            'budget': [1000000] * 6,
            'revenue': [5000000] * 6,
            'has_budget': [1] * 6,
            'has_revenue': [1] * 6,
            'has_runtime': [1] * 6,
            'is_adult': [0] * 6,
            'lang_en': [1] * 6,
            'is_independent': [0] * 6,
            'text_embedding': [np.random.normal(0, 0.1, 768).tolist() for _ in range(6)]
        })

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"
        sequences.to_parquet(sequences_path)
        movies.to_parquet(movies_path)

        try:
            data_loader = TwoTowerDataLoader(
                sequences_path=str(sequences_path),
                movies_path=str(movies_path),
                negative_sampling_ratio=2,
                min_user_interactions=1
            )

            # Should have positive examples
            assert len(data_loader.positive_examples) > 0
            # Should generate implicit negatives since no explicit negatives
            assert len(data_loader.negative_examples) > 0

        finally:
            shutil.rmtree(temp_dir)

    def test_all_negative_ratings(self):
        """Test handling when all ratings are negative."""
        temp_dir = tempfile.mkdtemp()

        # All negative ratings
        sequences = pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [1, 2, 3, 4, 5, 6],
            'rating': [1.0, 2.0, 1.5, 2.0, 1.0, 2.5],
            'thumbs_rating': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # All negative
            'datetime': pd.date_range('2020-01-01', periods=6, freq='D'),
            'sequence_id': ['1_seq_1'] * 2 + ['2_seq_1'] * 2 + ['3_seq_1'] * 2,
            'user_category': ['medium'] * 6
        })

        movies = pd.DataFrame({
            'movieId': list(range(1, 7)),
            'vote_average': [3.0] * 6,
            'vote_count': [50] * 6,
            'runtime': [90] * 6,
            'budget': [500000] * 6,
            'revenue': [1000000] * 6,
            'has_budget': [1] * 6,
            'has_revenue': [1] * 6,
            'has_runtime': [1] * 6,
            'is_adult': [0] * 6,
            'lang_en': [1] * 6,
            'is_independent': [0] * 6,
            'text_embedding': [np.random.normal(0, 0.1, 768).tolist() for _ in range(6)]
        })

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"
        sequences.to_parquet(sequences_path)
        movies.to_parquet(movies_path)

        try:
            data_loader = TwoTowerDataLoader(
                sequences_path=str(sequences_path),
                movies_path=str(movies_path),
                negative_sampling_ratio=1,
                min_user_interactions=1
            )

            # Should have negative examples
            assert len(data_loader.negative_examples) > 0
            # Should have no positive examples
            assert len(data_loader.positive_examples) == 0

        finally:
            shutil.rmtree(temp_dir)

    def test_mismatched_movie_ids(self):
        """Test handling when movie IDs don't match between sequences and features."""
        temp_dir = tempfile.mkdtemp()

        # Sequences reference movies 1-3
        sequences = pd.DataFrame({
            'userId': [1, 1, 2, 2],
            'movieId': [1, 2, 2, 3],  # Movies 1, 2, 3
            'rating': [4.0, 5.0, 3.0, 2.0],
            'thumbs_rating': [1.0, 2.0, 1.0, -1.0],
            'datetime': pd.date_range('2020-01-01', periods=4, freq='D'),
            'sequence_id': ['1_seq_1'] * 2 + ['2_seq_1'] * 2,
            'user_category': ['medium'] * 4
        })

        # Movies features only for movies 2, 3, 4 (missing movie 1)
        movies = pd.DataFrame({
            'movieId': [2, 3, 4],  # Missing movie 1, extra movie 4
            'vote_average': [7.0, 5.0, 8.0],
            'vote_count': [100, 50, 200],
            'runtime': [120, 90, 110],
            'budget': [1000000, 500000, 2000000],
            'revenue': [5000000, 1000000, 10000000],
            'has_budget': [1, 1, 1],
            'has_revenue': [1, 1, 1],
            'has_runtime': [1, 1, 1],
            'is_adult': [0, 0, 0],
            'lang_en': [1, 0, 1],
            'is_independent': [0, 0, 1],
            'text_embedding': [np.random.normal(0, 0.1, 768).tolist() for _ in range(3)]
        })

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"
        sequences.to_parquet(sequences_path)
        movies.to_parquet(movies_path)

        try:
            data_loader = TwoTowerDataLoader(
                sequences_path=str(sequences_path),
                movies_path=str(movies_path),
                negative_sampling_ratio=1,
                min_user_interactions=1
            )

            # Should only include sequences for movies that have features (2, 3)
            # Movie 1 should be filtered out
            valid_movie_ids = set(data_loader.sequences_df['movieId'].unique())
            assert 1 not in valid_movie_ids  # Movie 1 should be filtered out
            assert 2 in valid_movie_ids
            assert 3 in valid_movie_ids

        finally:
            shutil.rmtree(temp_dir)

    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        temp_dir = tempfile.mkdtemp()

        # Missing thumbs_rating column
        sequences_missing_thumbs = pd.DataFrame({
            'userId': [1, 1],
            'movieId': [1, 2],
            'rating': [4.0, 5.0],
            # 'thumbs_rating': missing!
            'datetime': pd.date_range('2020-01-01', periods=2, freq='D'),
            'sequence_id': ['1_seq_1'] * 2,
            'user_category': ['medium'] * 2
        })

        movies = pd.DataFrame({
            'movieId': [1, 2],
            'vote_average': [7.0, 8.0],
            'vote_count': [100, 200],
            'runtime': [120, 90],
            'budget': [1000000, 2000000],
            'revenue': [5000000, 10000000],
            'has_budget': [1, 1],
            'has_revenue': [1, 1],
            'has_runtime': [1, 1],
            'is_adult': [0, 0],
            'lang_en': [1, 1],
            'is_independent': [0, 1],
            'text_embedding': [np.random.normal(0, 0.1, 768).tolist() for _ in range(2)]
        })

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"
        sequences_missing_thumbs.to_parquet(sequences_path)
        movies.to_parquet(movies_path)

        try:
            with pytest.raises(KeyError):
                TwoTowerDataLoader(
                    sequences_path=str(sequences_path),
                    movies_path=str(movies_path),
                    negative_sampling_ratio=1,
                    min_user_interactions=1
                )
        finally:
            shutil.rmtree(temp_dir)

    def test_extremely_unbalanced_validation_split(self, minimal_valid_data):
        """Test handling of extremely unbalanced validation splits."""

        data_loader = TwoTowerDataLoader(
            sequences_path=minimal_valid_data['sequences_path'],
            movies_path=minimal_valid_data['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=1
        )

        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=4,
            user_hidden_dims=[4],
            item_hidden_dims=[4],
            dropout_rate=0.0
        )

        trainer = TwoTowerTrainer(model=model, data_loader=data_loader)

        # Test with very small validation split (might result in empty validation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history = trainer.train(num_epochs=1, batch_size=1, validation_split=0.05)

        # Should complete without crashing (division by zero protection)
        assert 'train_loss' in history
        assert 'val_loss' in history

    def test_zero_negative_sampling_ratio(self, minimal_valid_data):
        """Test with zero negative sampling ratio."""

        data_loader = TwoTowerDataLoader(
            sequences_path=minimal_valid_data['sequences_path'],
            movies_path=minimal_valid_data['movies_path'],
            negative_sampling_ratio=0,  # No implicit negatives
            min_user_interactions=1
        )

        # Should still have explicit negatives if they exist in data
        assert len(data_loader.positive_examples) > 0
        # May or may not have negatives depending on explicit ones in minimal data

    def test_very_high_negative_sampling_ratio(self, minimal_valid_data):
        """Test with very high negative sampling ratio."""

        data_loader = TwoTowerDataLoader(
            sequences_path=minimal_valid_data['sequences_path'],
            movies_path=minimal_valid_data['movies_path'],
            negative_sampling_ratio=10,  # Very high
            min_user_interactions=1
        )

        # Should handle high negative sampling without crashing
        assert len(data_loader.positive_examples) > 0
        assert len(data_loader.negative_examples) > 0
        # Negative examples should be significantly more than positive
        assert len(data_loader.negative_examples) >= len(data_loader.positive_examples)

    def test_single_movie_dataset(self):
        """Test with dataset containing only one movie."""
        temp_dir = tempfile.mkdtemp()

        # All sequences for the same movie
        sequences = pd.DataFrame({
            'userId': [1, 2, 3, 4],
            'movieId': [1, 1, 1, 1],  # Only one movie
            'rating': [4.0, 5.0, 2.0, 3.0],
            'thumbs_rating': [1.0, 2.0, -1.0, 1.0],
            'datetime': pd.date_range('2020-01-01', periods=4, freq='D'),
            'sequence_id': ['1_seq_1', '2_seq_1', '3_seq_1', '4_seq_1'],
            'user_category': ['medium'] * 4
        })

        movies = pd.DataFrame({
            'movieId': [1],
            'vote_average': [7.0],
            'vote_count': [100],
            'runtime': [120],
            'budget': [1000000],
            'revenue': [5000000],
            'has_budget': [1],
            'has_revenue': [1],
            'has_runtime': [1],
            'is_adult': [0],
            'lang_en': [1],
            'is_independent': [0],
            'text_embedding': [np.random.normal(0, 0.1, 768).tolist()]
        })

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"
        sequences.to_parquet(sequences_path)
        movies.to_parquet(movies_path)

        try:
            data_loader = TwoTowerDataLoader(
                sequences_path=str(sequences_path),
                movies_path=str(movies_path),
                negative_sampling_ratio=1,
                min_user_interactions=1
            )

            # Should handle single movie case
            assert data_loader.num_movies == 1
            assert len(data_loader.positive_examples) > 0

            # Note: Implicit negative sampling might be limited with only one movie
            # but should not crash

        finally:
            shutil.rmtree(temp_dir)

    def test_corrupted_embeddings_handling(self):
        """Test handling of corrupted or invalid text embeddings."""
        temp_dir = tempfile.mkdtemp()

        sequences = pd.DataFrame({
            'userId': [1, 1],
            'movieId': [1, 2],
            'rating': [4.0, 5.0],
            'thumbs_rating': [1.0, 2.0],
            'datetime': pd.date_range('2020-01-01', periods=2, freq='D'),
            'sequence_id': ['1_seq_1'] * 2,
            'user_category': ['medium'] * 2
        })

        # Movies with some corrupted embeddings
        movies = pd.DataFrame({
            'movieId': [1, 2],
            'vote_average': [7.0, 8.0],
            'vote_count': [100, 200],
            'runtime': [120, 90],
            'budget': [1000000, 2000000],
            'revenue': [5000000, 10000000],
            'has_budget': [1, 1],
            'has_revenue': [1, 1],
            'has_runtime': [1, 1],
            'is_adult': [0, 0],
            'lang_en': [1, 1],
            'is_independent': [0, 1],
            'text_embedding': [
                None,  # Corrupted embedding
                np.random.normal(0, 0.1, 768).tolist()  # Valid embedding
            ]
        })

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"
        sequences.to_parquet(sequences_path)
        movies.to_parquet(movies_path)

        try:
            data_loader = TwoTowerDataLoader(
                sequences_path=str(sequences_path),
                movies_path=str(movies_path),
                negative_sampling_ratio=1,
                min_user_interactions=1
            )

            # Should handle corrupted embeddings gracefully (replace with zeros)
            # The _prepare_movie_features method should handle None values
            assert data_loader.movie_features.shape[0] > 0

        finally:
            shutil.rmtree(temp_dir)

    def test_memory_efficiency_large_negative_sampling(self, minimal_valid_data):
        """Test memory efficiency with large negative sampling."""

        # This test ensures that large negative sampling doesn't cause memory issues
        data_loader = TwoTowerDataLoader(
            sequences_path=minimal_valid_data['sequences_path'],
            movies_path=minimal_valid_data['movies_path'],
            negative_sampling_ratio=100,  # Very large ratio
            min_user_interactions=1
        )

        # Should complete without memory errors
        assert len(data_loader.negative_examples) > len(data_loader.positive_examples)

        # Test that we can create datasets without memory issues
        dataset = TwoTowerDataset(
            data_loader,
            data_loader.positive_examples + data_loader.negative_examples
        )

        assert len(dataset) > 0