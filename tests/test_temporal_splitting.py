import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from movie_genie.retrieval.two_tower_model import TwoTowerDataLoader, TwoTowerTrainer, TwoTowerModel


class TestTemporalSplitting:
    """Test cases for temporal splitting functionality in the two-tower model."""

    @pytest.fixture
    def sample_temporal_sequences(self):
        """Create sample sequences with proper temporal ordering for testing."""
        base_time = datetime(2020, 1, 1)

        # Create sequences spanning 2 years with clear temporal patterns
        sequences = []
        for i in range(1000):
            user_id = (i % 100) + 1  # 100 unique users
            movie_id = (i % 200) + 1  # 200 unique movies

            # Create temporal progression
            days_offset = i // 10  # Every 10 interactions advance by 1 day
            timestamp = base_time + timedelta(days=days_offset)

            # Create different rating patterns over time for testing
            if i < 400:  # Early period: mix of all ratings
                thumbs_rating = np.random.choice([-1.0, 1.0, 2.0], p=[0.2, 0.6, 0.2])
            elif i < 700:  # Middle period: more positive ratings
                thumbs_rating = np.random.choice([-1.0, 1.0, 2.0], p=[0.1, 0.5, 0.4])
            else:  # Late period: different distribution
                thumbs_rating = np.random.choice([-1.0, 1.0, 2.0], p=[0.3, 0.4, 0.3])

            sequences.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': 3.0 + thumbs_rating,  # Original rating for reference
                'thumbs_rating': thumbs_rating,
                'datetime': timestamp,
                'sequence_id': f"{user_id}_seq_{i//20}",  # Group into sequences
                'user_category': 'medium'
            })

        return pd.DataFrame(sequences)

    @pytest.fixture
    def sample_movie_features(self):
        """Create sample movie features for testing."""
        movies = []
        for movie_id in range(1, 201):  # 200 movies
            # Create simple feature vectors
            text_embedding = np.random.normal(0, 1, 768).tolist()

            movies.append({
                'movieId': movie_id,
                'vote_average': np.random.uniform(1, 10),
                'vote_count': np.random.randint(1, 1000),
                'runtime': np.random.randint(80, 180),
                'budget': np.random.randint(1000000, 100000000),
                'revenue': np.random.randint(1000000, 500000000),
                'has_budget': 1,
                'has_revenue': 1,
                'has_runtime': 1,
                'is_adult': 0,
                'lang_en': 1,
                'lang_fr': 0,
                'is_independent': 0,
                'text_embedding': text_embedding
            })

        return pd.DataFrame(movies)

    @pytest.fixture
    def temp_files(self, sample_temporal_sequences, sample_movie_features):
        """Create temporary files for testing."""
        temp_dir = tempfile.mkdtemp()

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"

        sample_temporal_sequences.to_parquet(sequences_path)
        sample_movie_features.to_parquet(movies_path)

        yield {
            'sequences_path': str(sequences_path),
            'movies_path': str(movies_path),
            'temp_dir': temp_dir
        }

        shutil.rmtree(temp_dir)

    def test_create_temporal_splits_basic(self, sample_temporal_sequences):
        """Test basic temporal splitting functionality."""
        # Create a mock trainer instance to access the method
        from unittest.mock import MagicMock
        import torch.nn as nn

        # Create a minimal mock model with at least one parameter
        mock_model = nn.Linear(1, 1)  # Simple model with parameters
        trainer = TwoTowerTrainer(model=mock_model, data_loader=None)

        train_df, val_df, test_df = trainer.create_temporal_splits(
            sample_temporal_sequences,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Check split sizes
        total_size = len(sample_temporal_sequences)
        assert len(train_df) == int(total_size * 0.6)
        assert len(val_df) == int(total_size * 0.2)
        assert len(test_df) >= int(total_size * 0.2) - 1  # Allow for rounding

        # Check temporal ordering
        assert train_df['datetime'].max() <= val_df['datetime'].min()
        assert val_df['datetime'].max() <= test_df['datetime'].min()

        # Check all data is preserved
        assert len(train_df) + len(val_df) + len(test_df) == total_size

    def test_create_temporal_splits_with_timestamp_column(self, sample_temporal_sequences):
        """Test temporal splitting when using 'timestamp' column instead of 'datetime'."""
        # Convert datetime to timestamp
        sequences_with_timestamp = sample_temporal_sequences.copy()
        sequences_with_timestamp['timestamp'] = sequences_with_timestamp['datetime'].astype(int) // 10**9
        sequences_with_timestamp = sequences_with_timestamp.drop('datetime', axis=1)

        import torch.nn as nn
        # Create a minimal mock model with parameters
        mock_model = nn.Linear(1, 1)
        trainer = TwoTowerTrainer(model=mock_model, data_loader=None)

        train_df, val_df, test_df = trainer.create_temporal_splits(
            sequences_with_timestamp,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )

        # Check temporal ordering with timestamp
        assert train_df['timestamp'].max() <= val_df['timestamp'].min()
        assert val_df['timestamp'].max() <= test_df['timestamp'].min()

    def test_create_temporal_splits_no_timestamp(self, sample_temporal_sequences):
        """Test temporal splitting fallback when no timestamp columns exist."""
        # Remove both datetime and timestamp columns
        sequences_no_time = sample_temporal_sequences.drop('datetime', axis=1)

        import torch.nn as nn
        # Create a minimal mock model with parameters
        mock_model = nn.Linear(1, 1)
        trainer = TwoTowerTrainer(model=mock_model, data_loader=None)

        # Should work even without timestamp columns (logs warning but doesn't raise)
        train_df, val_df, test_df = trainer.create_temporal_splits(
            sequences_no_time,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Should still work using row order
        total_size = len(sequences_no_time)
        assert len(train_df) == int(total_size * 0.6)
        assert len(val_df) == int(total_size * 0.2)

    def test_create_examples_from_sequences(self, temp_files):
        """Test example creation from sequences with proper thumbs ratings."""
        data_loader = TwoTowerDataLoader(
            sequences_path=temp_files['sequences_path'],
            movies_path=temp_files['movies_path'],
            negative_sampling_ratio=2,
            min_user_interactions=3
        )

        # Test example creation from full sequences
        positive_examples, negative_examples = data_loader._create_examples_from_sequences(
            data_loader.sequences_df, generate_implicit=True
        )

        # Check that examples were created
        assert len(positive_examples) > 0
        assert len(negative_examples) > 0

        # Check example structure
        assert all(key in positive_examples[0] for key in ['user_idx', 'movie_idx', 'rating'])
        assert all(key in negative_examples[0] for key in ['user_idx', 'movie_idx', 'rating'])

        # Check rating values
        pos_ratings = [ex['rating'] for ex in positive_examples]
        assert all(rating >= 1.0 for rating in pos_ratings)  # Positive examples

        neg_ratings = [ex['rating'] for ex in negative_examples]
        explicit_negs = [rating for rating in neg_ratings if rating == -1.0]
        implicit_negs = [rating for rating in neg_ratings if rating == 0.0]
        assert len(explicit_negs) > 0  # Should have explicit negatives
        assert len(implicit_negs) > 0  # Should have implicit negatives

    def test_create_examples_without_implicit(self, temp_files):
        """Test example creation without implicit negative generation."""
        data_loader = TwoTowerDataLoader(
            sequences_path=temp_files['sequences_path'],
            movies_path=temp_files['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=3
        )

        # Test with small subset to ensure no implicit negatives needed
        small_sequences = data_loader.sequences_df.head(100)

        positive_examples, negative_examples = data_loader._create_examples_from_sequences(
            small_sequences, generate_implicit=False
        )

        # Check that no implicit negatives were generated
        neg_ratings = [ex['rating'] for ex in negative_examples]
        implicit_negs = [rating for rating in neg_ratings if rating == 0.0]
        assert len(implicit_negs) == 0  # No implicit negatives should be generated

    def test_temporal_train_val_split_integration(self, temp_files):
        """Test the complete temporal train-validation split integration."""
        data_loader = TwoTowerDataLoader(
            sequences_path=temp_files['sequences_path'],
            movies_path=temp_files['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=3
        )

        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=32,
            user_hidden_dims=[32],
            item_hidden_dims=[32],
            dropout_rate=0.1
        )

        trainer = TwoTowerTrainer(
            model=model,
            data_loader=data_loader,
            learning_rate=0.001,
            margin=1.0
        )

        # Test the temporal splitting
        train_examples, val_examples = trainer._create_train_val_split(validation_split=0.2)

        # Check that both splits have positive and negative examples
        assert len(train_examples['positive']) > 0
        assert len(train_examples['negative']) > 0
        assert len(val_examples['positive']) > 0
        assert len(val_examples['negative']) > 0

        # Check reasonable ratios
        train_total = len(train_examples['positive']) + len(train_examples['negative'])
        val_total = len(val_examples['positive']) + len(val_examples['negative'])

        train_pos_ratio = len(train_examples['positive']) / train_total
        val_pos_ratio = len(val_examples['positive']) / val_total

        # Both should have reasonable positive ratios (not 0 or 1)
        assert 0.1 < train_pos_ratio < 0.9
        assert 0.1 < val_pos_ratio < 0.9

    def test_analyze_split_overlap(self, sample_temporal_sequences):
        """Test the split overlap analysis functionality."""
        import torch.nn as nn
        # Create a minimal mock model with parameters
        mock_model = nn.Linear(1, 1)
        trainer = TwoTowerTrainer(model=mock_model, data_loader=None)

        train_df, val_df, test_df = trainer.create_temporal_splits(
            sample_temporal_sequences,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Test overlap analysis
        analysis = trainer.analyze_split_overlap(train_df, val_df, test_df)

        # Check analysis structure
        assert 'user_stats' in analysis
        assert 'item_stats' in analysis
        assert 'user_overlap_pct' in analysis
        assert 'item_overlap_pct' in analysis

        # Check that stats are reasonable
        assert analysis['user_stats']['train_users'] > 0
        assert analysis['user_stats']['val_users'] > 0
        assert analysis['item_stats']['train_items'] > 0
        assert analysis['item_stats']['val_items'] > 0

        # Check that percentages are between 0 and 100
        for pct in analysis['user_overlap_pct'].values():
            assert 0 <= pct <= 100
        for pct in analysis['item_overlap_pct'].values():
            assert 0 <= pct <= 100

    def test_load_and_split_data(self, temp_files):
        """Test the complete data loading and splitting pipeline."""
        import torch.nn as nn
        # Create a minimal mock model with parameters
        mock_model = nn.Linear(1, 1)
        trainer = TwoTowerTrainer(model=mock_model, data_loader=None)

        train_df, val_df, test_df, analysis = trainer.load_and_split_data(
            sequences_path=temp_files['sequences_path'],
            movies_path=temp_files['movies_path'],
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Check that all components are returned
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(analysis, dict)

        # Check split sizes
        total_expected = len(train_df) + len(val_df) + len(test_df)
        assert total_expected == 1000  # Our sample size

        # Check that analysis contains expected information
        assert 'user_stats' in analysis
        assert 'item_stats' in analysis

    def test_invalid_split_ratios(self, sample_temporal_sequences):
        """Test that invalid split ratios raise appropriate errors."""
        import torch.nn as nn
        # Create a minimal mock model with parameters
        mock_model = nn.Linear(1, 1)
        trainer = TwoTowerTrainer(model=mock_model, data_loader=None)

        # Test ratios that don't sum to 1.0
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            trainer.load_and_split_data(
                sequences_path="dummy_path",
                movies_path="dummy_path",
                train_ratio=0.6,
                val_ratio=0.3,
                test_ratio=0.3  # Sum = 1.2
            )

    def test_empty_sequences_handling(self, temp_files):
        """Test handling of empty sequences DataFrame."""
        # Create empty sequences file
        empty_sequences = pd.DataFrame(columns=['userId', 'movieId', 'thumbs_rating', 'datetime'])
        empty_path = Path(temp_files['temp_dir']) / "empty_sequences.parquet"
        empty_sequences.to_parquet(empty_path)

        data_loader = TwoTowerDataLoader(
            sequences_path=str(empty_path),
            movies_path=temp_files['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=1
        )

        # Should handle empty data gracefully
        positive_examples, negative_examples = data_loader._create_examples_from_sequences(
            data_loader.sequences_df, generate_implicit=False
        )

        assert len(positive_examples) == 0
        assert len(negative_examples) == 0

    def test_single_user_sequences(self, temp_files, sample_movie_features):
        """Test temporal splitting with sequences from a single user."""
        # Create sequences for only one user
        single_user_sequences = pd.DataFrame({
            'userId': [1] * 50,
            'movieId': range(1, 51),
            'rating': [4.0] * 50,
            'thumbs_rating': [1.0] * 50,
            'datetime': pd.date_range('2020-01-01', periods=50, freq='D'),
            'sequence_id': ['1_seq_1'] * 50,
            'user_category': ['medium'] * 50
        })

        single_user_path = Path(temp_files['temp_dir']) / "single_user.parquet"
        single_user_sequences.to_parquet(single_user_path)

        data_loader = TwoTowerDataLoader(
            sequences_path=str(single_user_path),
            movies_path=temp_files['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=1
        )

        # Should work even with single user
        positive_examples, negative_examples = data_loader._create_examples_from_sequences(
            data_loader.sequences_df, generate_implicit=True
        )

        # Should have positive examples from the single user
        assert len(positive_examples) > 0
        # May have implicit negatives generated
        assert len(negative_examples) >= 0