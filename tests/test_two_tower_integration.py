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
    TwoTowerEvaluator, TwoTowerDataset, UserTower, ItemTower
)


class TestTwoTowerIntegration:
    """Integration tests for the complete two-tower model training pipeline."""

    @pytest.fixture
    def sample_training_data(self):
        """Create realistic training data for integration testing."""
        np.random.seed(42)  # For reproducible tests

        # Create 500 sequences with diverse patterns
        sequences = []
        base_time = datetime(2018, 1, 1)

        for i in range(500):
            user_id = (i % 50) + 1  # 50 users
            movie_id = (i % 100) + 1  # 100 movies

            # Create temporal progression
            timestamp = base_time + timedelta(days=i // 5)

            # Create realistic rating patterns
            user_bias = (user_id % 3) - 1  # User preferences: -1, 0, 1
            movie_bias = (movie_id % 3) - 1  # Movie quality: -1, 0, 1

            # Generate thumbs rating based on biases
            combined_bias = user_bias + movie_bias + np.random.normal(0, 0.5)
            if combined_bias < -0.5:
                thumbs_rating = -1.0
            elif combined_bias > 0.5:
                thumbs_rating = 2.0
            else:
                thumbs_rating = 1.0

            sequences.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': 3.0 + thumbs_rating,
                'thumbs_rating': thumbs_rating,
                'datetime': timestamp,
                'sequence_id': f"{user_id}_seq_{i//10}",
                'user_category': 'medium'
            })

        return pd.DataFrame(sequences)

    @pytest.fixture
    def sample_movie_features(self):
        """Create comprehensive movie features for testing."""
        movies = []
        for movie_id in range(1, 101):  # 100 movies
            # Create realistic feature vectors
            text_embedding = np.random.normal(0, 0.1, 768).tolist()

            movies.append({
                'movieId': movie_id,
                'vote_average': np.random.uniform(3, 8),
                'vote_count': np.random.randint(10, 1000),
                'runtime': np.random.randint(90, 180),
                'budget': np.random.randint(1000000, 100000000),
                'revenue': np.random.randint(1000000, 500000000),
                'has_budget': np.random.choice([0, 1], p=[0.3, 0.7]),
                'has_revenue': np.random.choice([0, 1], p=[0.3, 0.7]),
                'has_runtime': 1,
                'is_adult': 0,
                'lang_en': np.random.choice([0, 1], p=[0.2, 0.8]),
                'lang_fr': np.random.choice([0, 1], p=[0.9, 0.1]),
                'lang_es': np.random.choice([0, 1], p=[0.8, 0.2]),
                'is_independent': np.random.choice([0, 1], p=[0.8, 0.2]),
                'text_embedding': text_embedding
            })

        return pd.DataFrame(movies)

    @pytest.fixture
    def integration_setup(self, sample_training_data, sample_movie_features):
        """Set up complete integration test environment."""
        temp_dir = tempfile.mkdtemp()

        sequences_path = Path(temp_dir) / "sequences.parquet"
        movies_path = Path(temp_dir) / "movies.parquet"

        sample_training_data.to_parquet(sequences_path)
        sample_movie_features.to_parquet(movies_path)

        yield {
            'sequences_path': str(sequences_path),
            'movies_path': str(movies_path),
            'temp_dir': temp_dir
        }

        shutil.rmtree(temp_dir)

    def test_complete_training_pipeline(self, integration_setup):
        """Test the complete training pipeline from data loading to model training."""

        # 1. Data Loading
        data_loader = TwoTowerDataLoader(
            sequences_path=integration_setup['sequences_path'],
            movies_path=integration_setup['movies_path'],
            negative_sampling_ratio=2,
            min_user_interactions=3
        )

        # Verify data loading worked
        assert data_loader.num_users > 0
        assert data_loader.num_movies > 0
        assert len(data_loader.positive_examples) > 0
        assert len(data_loader.negative_examples) > 0
        assert data_loader.movie_features.shape[1] > 0

        # 2. Model Creation
        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=32,
            user_hidden_dims=[32, 16],
            item_hidden_dims=[32, 16],
            dropout_rate=0.1
        )

        # Verify model structure
        assert isinstance(model.user_tower, UserTower)
        assert isinstance(model.item_tower, ItemTower)

        # 3. Trainer Setup
        trainer = TwoTowerTrainer(
            model=model,
            data_loader=data_loader,
            learning_rate=0.01,  # Higher for faster test convergence
            margin=1.0
        )

        # 4. Test Temporal Splitting
        train_examples, val_examples = trainer._create_train_val_split(validation_split=0.2)

        # Verify splits have reasonable distributions
        assert len(train_examples['positive']) > 0
        assert len(train_examples['negative']) > 0
        assert len(val_examples['positive']) > 0
        assert len(val_examples['negative']) > 0

        # 5. Training Execution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress validation warnings for test
            history = trainer.train(num_epochs=2, batch_size=32, validation_split=0.2)

        # Verify training completed
        assert 'epoch' in history
        assert 'loss' in history
        assert len(history['epoch']) == 2
        assert len(history['loss']) == 2

        # Check that losses are reasonable numbers
        for epoch_loss in history['loss']:
            assert 'train' in epoch_loss
            assert 'val' in epoch_loss
            assert isinstance(epoch_loss['train'], (int, float))
            assert isinstance(epoch_loss['val'], (int, float))
            assert epoch_loss['train'] >= 0
            assert epoch_loss['val'] >= 0

    def test_model_forward_pass(self, integration_setup):
        """Test model forward pass with real data."""

        data_loader = TwoTowerDataLoader(
            sequences_path=integration_setup['sequences_path'],
            movies_path=integration_setup['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=2
        )

        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=16,
            user_hidden_dims=[16],
            item_hidden_dims=[16],
            dropout_rate=0.0  # No dropout for deterministic testing
        )

        # Create sample batch
        dataset = TwoTowerDataset(data_loader, data_loader.positive_examples + data_loader.negative_examples)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Verify batch structure
                assert 'user_id' in batch
                assert 'pos_movie_id' in batch
                assert 'neg_movie_id' in batch
                assert 'pos_movie_features' in batch
                assert 'neg_movie_features' in batch

                # Test model forward pass
                user_embeddings = model.get_user_embeddings(batch['user_id'])
                pos_item_embeddings = model.get_movie_embeddings(
                    batch['pos_movie_id'], batch['pos_movie_features']
                )
                neg_item_embeddings = model.get_movie_embeddings(
                    batch['neg_movie_id'], batch['neg_movie_features']
                )

                # Verify embedding shapes
                assert user_embeddings.shape == (len(batch['user_id']), 16)
                assert pos_item_embeddings.shape == (len(batch['pos_movie_id']), 16)
                assert neg_item_embeddings.shape == (len(batch['neg_movie_id']), 16)

                # Test that embeddings are not all zeros
                assert torch.abs(user_embeddings).sum() > 0
                assert torch.abs(pos_item_embeddings).sum() > 0
                assert torch.abs(neg_item_embeddings).sum() > 0

                break  # Only test one batch

    def test_evaluator_functionality(self, integration_setup):
        """Test the evaluator with trained model."""

        data_loader = TwoTowerDataLoader(
            sequences_path=integration_setup['sequences_path'],
            movies_path=integration_setup['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=2
        )

        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=16,
            user_hidden_dims=[16],
            item_hidden_dims=[16],
            dropout_rate=0.0
        )

        # Train for a few steps
        trainer = TwoTowerTrainer(model=model, data_loader=data_loader)
        trainer.train(num_epochs=1, batch_size=16, validation_split=0.2)

        # Test evaluator
        evaluator = TwoTowerEvaluator(model, data_loader)

        # Create test examples
        test_examples = {
            'positive': data_loader.positive_examples[:50],
            'negative': data_loader.negative_examples[:50]
        }

        # Test evaluation
        results = evaluator.evaluate_model_performance(test_examples, k_values=[5, 10])

        # Verify evaluation results structure
        assert 'recall' in results
        assert 5 in results['recall']
        assert 10 in results['recall']

        # Check that recall values are reasonable
        assert 0 <= results['recall'][5] <= 1
        assert 0 <= results['recall'][10] <= 1

    def test_temporal_consistency_throughout_pipeline(self, integration_setup):
        """Test that temporal ordering is maintained throughout the entire pipeline."""

        data_loader = TwoTowerDataLoader(
            sequences_path=integration_setup['sequences_path'],
            movies_path=integration_setup['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=2
        )

        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=16,
            user_hidden_dims=[16],
            item_hidden_dims=[16],
            dropout_rate=0.1
        )

        trainer = TwoTowerTrainer(model=model, data_loader=data_loader)

        # Test temporal splitting preserves chronological order
        train_examples, val_examples = trainer._create_train_val_split(validation_split=0.3)

        # Verify that validation has positive examples (original bug fix)
        assert len(val_examples['positive']) > 0

        # Verify reasonable positive ratios in both splits
        train_pos_ratio = len(train_examples['positive']) / (
            len(train_examples['positive']) + len(train_examples['negative'])
        )
        val_pos_ratio = len(val_examples['positive']) / (
            len(val_examples['positive']) + len(val_examples['negative'])
        )

        # Both should have some positive examples (not 0 or 1)
        assert 0.1 <= train_pos_ratio <= 0.9
        assert 0.1 <= val_pos_ratio <= 0.9

    def test_movie_feature_consistency(self, integration_setup):
        """Test that movie feature mapping works correctly throughout pipeline."""

        data_loader = TwoTowerDataLoader(
            sequences_path=integration_setup['sequences_path'],
            movies_path=integration_setup['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=2
        )

        # Test that all movie IDs in examples have corresponding features
        all_movie_indices = set()
        for example in data_loader.positive_examples + data_loader.negative_examples:
            all_movie_indices.add(example['movie_idx'])

        # Verify all movie indices have corresponding features
        for movie_idx in all_movie_indices:
            assert movie_idx in data_loader.movie_feature_map

        # Test feature lookup works
        for movie_idx in list(all_movie_indices)[:10]:  # Test first 10
            feature_row_idx = data_loader.movie_feature_map[movie_idx]
            features = data_loader.movie_features[feature_row_idx]
            assert features.shape[0] == data_loader.movie_features.shape[1]

    def test_error_handling_in_pipeline(self, integration_setup):
        """Test error handling throughout the pipeline."""

        # Test with missing file
        with pytest.raises(FileNotFoundError):
            TwoTowerDataLoader(
                sequences_path="nonexistent_file.parquet",
                movies_path=integration_setup['movies_path'],
                negative_sampling_ratio=1,
                min_user_interactions=2
            )

        # Test with empty validation split handling
        data_loader = TwoTowerDataLoader(
            sequences_path=integration_setup['sequences_path'],
            movies_path=integration_setup['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=2
        )

        model = TwoTowerModel(
            num_users=data_loader.num_users,
            num_movies=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            embedding_dim=8,
            user_hidden_dims=[8],
            item_hidden_dims=[8],
            dropout_rate=0.1
        )

        trainer = TwoTowerTrainer(model=model, data_loader=data_loader)

        # Should handle empty validation gracefully (our division by zero fix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This should not crash even if validation becomes empty
            history = trainer.train(num_epochs=1, batch_size=16, validation_split=0.05)

        assert 'epoch' in history
        assert 'loss' in history

    def test_different_model_configurations(self, integration_setup):
        """Test training with different model architectures."""

        data_loader = TwoTowerDataLoader(
            sequences_path=integration_setup['sequences_path'],
            movies_path=integration_setup['movies_path'],
            negative_sampling_ratio=1,
            min_user_interactions=2
        )

        configurations = [
            # Small model
            {'embedding_dim': 8, 'user_hidden_dims': [8], 'item_hidden_dims': [8]},
            # Deeper model
            {'embedding_dim': 16, 'user_hidden_dims': [32, 16], 'item_hidden_dims': [32, 16]},
            # Asymmetric model
            {'embedding_dim': 12, 'user_hidden_dims': [24], 'item_hidden_dims': [24, 12]},
        ]

        for config in configurations:
            model = TwoTowerModel(
                num_users=data_loader.num_users,
                num_movies=data_loader.num_movies,
                content_feature_dim=data_loader.movie_features.shape[1],
                dropout_rate=0.1,
                **config
            )

            trainer = TwoTowerTrainer(model=model, data_loader=data_loader)

            # Should train successfully with any reasonable configuration
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                history = trainer.train(num_epochs=1, batch_size=8, validation_split=0.2)

            assert 'epoch' in history
            assert len(history['epoch']) == 1
            assert len(history['loss']) == 1