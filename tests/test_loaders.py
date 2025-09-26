import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from movie_genie.data.loaders import MovieLensLoader


class TestMovieLensLoader:
    @pytest.fixture
    def sample_ratings_data(self):
        """Create sample ratings data for testing"""
        return pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            'movieId': [1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9],
            'rating': [4.0, 3.5, 5.0, 2.5, 4.5, 3.0, 2.0, 4.0, 3.5, 5.0, 4.5, 2.5, 3.0, 4.5, 2.0, 3.5],
            'timestamp': [1000000000, 1000086400, 1000172800, 1000259200, 
                         1000345600, 1000432000, 1000518400, 1000604800,
                         1000691200, 1000777600, 1000864000, 1000950400,
                         1001036800, 1001123200, 1001209600, 1001296000]
        })

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def loader_with_temp_dir(self, temp_dir):
        """Create loader with temporary directory"""
        with patch('yaml.safe_load') as mock_yaml, patch('builtins.open'):
            mock_yaml.return_value = {
                'data_sources': {
                    'movielens': {
                        'dataset_size': 'test',
                        'cache_dir': 'data/processed'
                    }
                }
            }
            loader = MovieLensLoader("dummy_config.yaml")
            loader.data_path = Path(temp_dir)  # Override for testing
            return loader

    @patch('yaml.safe_load')
    @patch('builtins.open')
    def test_init(self, mock_open, mock_yaml, temp_dir):
        """Test MovieLensLoader initialization"""
        mock_yaml.return_value = {
            'data_sources': {
                'movielens': {
                    'dataset_size': 'ml-100k',
                    'cache_dir': 'data/processed'
                }
            }
        }
        
        loader = MovieLensLoader("dummy_config.yaml")
        assert loader.data_path == Path("data/raw/ml-100k")
        assert loader.ratings_cache is None
        assert loader.movies_cache is None

    def test_load_ratings_file_not_found(self, loader_with_temp_dir):
        """Test load_ratings raises FileNotFoundError when file doesn't exist"""
        with pytest.raises(FileNotFoundError, match="Ratings file not found"):
            loader_with_temp_dir.load_ratings()

    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    @patch('yaml.safe_load')
    @patch('builtins.open')
    def test_load_ratings_success(self, mock_file_open, mock_yaml, mock_exists, mock_read_csv, sample_ratings_data):
        """Test successful loading of ratings file"""
        mock_yaml.return_value = {
            'data_sources': {
                'movielens': {
                    'dataset_size': 'test',
                    'cache_dir': 'data/processed'
                }
            }
        }
        mock_exists.return_value = True
        
        # Mock read_csv to return sample data with correct dtypes
        expected_data = sample_ratings_data.copy()
        expected_data = expected_data.astype({
            'userId': 'int32',
            'movieId': 'int32', 
            'rating': 'float32',
            'timestamp': 'int64'
        })
        mock_read_csv.return_value = expected_data
        
        loader = MovieLensLoader("dummy_config.yaml")
        result = loader.load_ratings()
        
        # Check data types
        assert result['userId'].dtype == 'int32'
        assert result['movieId'].dtype == 'int32'
        assert result['rating'].dtype == 'float32'
        assert result['timestamp'].dtype == 'int64'
        
        # Check data integrity
        assert len(result) == len(sample_ratings_data)
        assert loader.ratings_cache is not None

    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    @patch('yaml.safe_load')
    @patch('builtins.open')
    def test_load_ratings_caching(self, mock_file_open, mock_yaml, mock_exists, mock_read_csv, sample_ratings_data):
        """Test that ratings are cached after first load"""
        mock_yaml.return_value = {
            'data_sources': {
                'movielens': {
                    'dataset_size': 'test',
                    'cache_dir': 'data/processed'
                }
            }
        }
        mock_exists.return_value = True
        
        # Mock read_csv to return sample data
        expected_data = sample_ratings_data.copy()
        expected_data = expected_data.astype({
            'userId': 'int32',
            'movieId': 'int32', 
            'rating': 'float32',
            'timestamp': 'int64'
        })
        mock_read_csv.return_value = expected_data
        
        loader = MovieLensLoader("dummy_config.yaml")
        
        # First load
        result1 = loader.load_ratings()
        assert loader.ratings_cache is not None
        
        # Second load should return cached data (read_csv should only be called once)
        result2 = loader.load_ratings()
        assert result1 is result2  # Same object reference
        assert mock_read_csv.call_count == 1  # Called only once due to caching

    def test_categorize_users_by_activity_basic(self):
        """Test user categorization by activity"""
        test_data = pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            'days_since_last': [np.nan, 10, np.nan, 20, np.nan, 30, np.nan, 40, 
                               np.nan, 50, np.nan, 60, np.nan, 70, np.nan, 80]
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        result = loader.categorize_users_by_activity(test_data)
        
        # Check basic structure
        assert len(result) == 8
        assert set(result.columns) == {'userId', 'user_category'}
        
        # Check that all categories are valid
        valid_categories = {'heavy', 'moderate_high', 'moderate_low', 'casual', 'single_rating'}
        assert all(cat in valid_categories for cat in result['user_category'])

    def test_categorize_users_single_rating(self):
        """Test categorization handles users with single ratings"""
        test_data = pd.DataFrame({
            'userId': [1, 2, 2],
            'days_since_last': [np.nan, np.nan, 10]
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        result = loader.categorize_users_by_activity(test_data)
        
        # User 1 should be 'single_rating' (only NaN values)
        user1_category = result[result['userId'] == 1]['user_category'].iloc[0]
        assert user1_category == 'single_rating'

    def test_calculate_adaptive_windows(self):
        """Test adaptive windows calculation"""
        user_categories = pd.DataFrame({
            'userId': [1, 2, 3],
            'user_category': ['heavy', 'moderate_high', 'casual']
        })
        
        user_gaps = pd.Series([10, 20, 30], index=[1, 2, 3], name='days_since_last')
        user_gaps.index.name = 'userId'
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        result = loader.calculate_adaptive_windows(user_categories, user_gaps)
        
        # Check structure
        expected_columns = {'userId', 'user_category', 'median_gap_days', 'sequence_window_days'}
        assert set(result.columns) == expected_columns
        
        # Check 2.5x multiplier
        assert result['sequence_window_days'].iloc[0] == 25.0  # 10 * 2.5
        assert result['sequence_window_days'].iloc[1] == 50.0  # 20 * 2.5
        assert result['sequence_window_days'].iloc[2] == 75.0  # 30 * 2.5

    def test_analyze_user_rating_patterns(self):
        """Test user rating pattern analysis"""
        test_data = pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [1, 2, 3, 4, 5, 6],
            'rating': [4.0, 3.5, 5.0, 2.5, 4.5, 3.0],
            'timestamp': [1000000000, 1000086400, 1000172800, 1000259200, 1000345600, 1000432000]
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        processed_df, result = loader.analyze_user_rating_patterns(test_data)
        
        # Check processed dataframe has new columns
        assert 'datetime' in processed_df.columns
        assert 'days_since_last' in processed_df.columns
        
        # Check return structure
        expected_keys = {'median_user_ratings', 'median_rating_gap_days', 
                        'heavy_users_threshold', 'recommended_sequence_window'}
        assert set(result.keys()) == expected_keys
        
        # Check types
        assert isinstance(result['median_user_ratings'], (int, float))
        assert isinstance(result['median_rating_gap_days'], (int, float))
        assert isinstance(result['heavy_users_threshold'], (int, float))
        assert isinstance(result['recommended_sequence_window'], int)

    def test_create_adaptive_sequences(self):
        """Test adaptive sequence creation"""
        # Create test data with datetime
        test_data = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2],
            'movieId': [1, 2, 3, 4, 5],
            'rating': [4.0, 3.5, 5.0, 2.5, 4.5],
            'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-10', 
                                      '2023-01-01', '2023-01-05'])
        })
        
        user_windows = pd.DataFrame({
            'userId': [1, 2],
            'sequence_window_days': [5.0, 3.0],
            'user_category': ['heavy', 'moderate_high']
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        result = loader.create_adaptive_sequences(test_data, user_windows)
        
        # Check structure
        expected_columns = {'userId', 'movieId', 'rating', 'datetime', 'sequence_id', 'user_category'}
        assert set(result.columns) == expected_columns
        
        # Check sequence IDs are created
        assert all(result['sequence_id'].str.contains('_seq_'))

    @patch('pandas.DataFrame.to_parquet')
    @patch('pathlib.Path.mkdir')
    def test_save_processed_sequences(self, mock_mkdir, mock_to_parquet):
        """Test saving processed sequences"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        user_windows = pd.DataFrame({'col2': [4, 5, 6]})
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        loader.save_processed_sequences(df, user_windows, "test_cache")
        
        # Check mkdir was called
        mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)
        
        # Check to_parquet was called twice
        assert mock_to_parquet.call_count == 2

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.exists')
    def test_load_or_process_sequences_cached(self, mock_exists, mock_read_parquet):
        """Test loading from cache when files exist"""
        mock_exists.return_value = True
        mock_read_parquet.side_effect = [
            pd.DataFrame({'sequences': [1, 2, 3]}),
            pd.DataFrame({'windows': [4, 5, 6]})
        ]
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        sequences, windows = loader.load_or_process_sequences(force_reprocess=False)
        
        assert mock_read_parquet.call_count == 2
        assert 'sequences' in sequences.columns
        assert 'windows' in windows.columns

    @patch('pandas.read_parquet')
    @patch('pathlib.Path.exists')
    def test_load_or_process_sequences_force_reprocess(self, mock_exists, mock_read_parquet):
        """Test forcing reprocessing even when cache exists"""
        mock_exists.return_value = True
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        
        # Mock the load_ratings method to avoid file I/O
        sample_data = pd.DataFrame({
            'userId': [1, 1, 2, 2],
            'movieId': [1, 2, 3, 4], 
            'rating': [4.0, 3.5, 5.0, 2.5],
            'timestamp': [1000000000, 1000086400, 1000172800, 1000259200]
        })
        
        # Mock analyze_user_rating_patterns to return tuple (DataFrame, dict)
        def mock_analyze_patterns(df):
            df_copy = df.copy()
            df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
            df_copy = df_copy.sort_values(['userId', 'datetime'])
            df_copy['days_since_last'] = df_copy.groupby('userId')['datetime'].diff().dt.days
            
            analysis_results = {
                'median_user_ratings': 2.0,
                'median_rating_gap_days': 1.0,
                'heavy_users_threshold': 2.0,
                'recommended_sequence_window': 3
            }
            return df_copy, analysis_results
        
        with patch.object(loader, 'load_ratings', return_value=sample_data), \
             patch.object(loader, 'analyze_user_rating_patterns', side_effect=mock_analyze_patterns), \
             patch.object(loader, 'save_processed_sequences'):
            sequences, windows = loader.load_or_process_sequences(force_reprocess=True)
        
        # Should not read from parquet when force_reprocess=True
        mock_read_parquet.assert_not_called()
        assert 'userId' in sequences.columns
        assert 'userId' in windows.columns

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame(columns=['userId', 'days_since_last'])
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        
        # Should handle empty dataframe gracefully - returns empty result
        result = loader.categorize_users_by_activity(empty_df)
        assert len(result) == 0
        assert set(result.columns) == {'userId', 'user_category'}

    def test_analyze_user_rating_patterns_edge_cases(self):
        """Test analyze_user_rating_patterns with edge cases"""
        # Single user with single rating - this will cause NaN median_gap_days
        single_rating_data = pd.DataFrame({
            'userId': [1],
            'movieId': [1],
            'rating': [4.0],
            'timestamp': [1000000000]
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        
        # This should raise ValueError due to NaN conversion to int
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            loader.analyze_user_rating_patterns(single_rating_data)

    def test_calculate_adaptive_windows_edge_cases(self):
        """Test calculate_adaptive_windows with edge cases"""
        # Users with zero gaps (single ratings)
        user_categories = pd.DataFrame({
            'userId': [1, 2],
            'user_category': ['single_rating', 'heavy']
        })
        
        user_gaps = pd.Series([np.nan, 5.0], index=[1, 2], name='days_since_last')
        user_gaps.index.name = 'userId'
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        
        result = loader.calculate_adaptive_windows(user_categories, user_gaps)
        
        # User 1 should have NaN for median_gap_days and sequence_window_days
        user1_row = result[result['userId'] == 1]
        assert pd.isna(user1_row['median_gap_days'].iloc[0])
        assert pd.isna(user1_row['sequence_window_days'].iloc[0])
        
        # User 2 should have proper calculations
        user2_row = result[result['userId'] == 2]
        assert user2_row['median_gap_days'].iloc[0] == 5.0
        assert user2_row['sequence_window_days'].iloc[0] == 12.5

    def test_create_adaptive_sequences_single_user(self):
        """Test create_adaptive_sequences with single user data"""
        test_data = pd.DataFrame({
            'userId': [1, 1, 1],
            'movieId': [1, 2, 3],
            'rating': [4.0, 3.5, 5.0],
            'datetime': pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-10'])
        })
        
        user_windows = pd.DataFrame({
            'userId': [1],
            'sequence_window_days': [5.0],
            'user_category': ['moderate_high']
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        
        result = loader.create_adaptive_sequences(test_data, user_windows)
        
        # Should create multiple sequences based on gaps > 5 days
        sequence_ids = result['sequence_id'].unique()
        assert len(sequence_ids) >= 2  # At least 2 sequences due to the 7-day gap

    def test_convert_to_thumbs_ratings(self):
        """Test conversion of 5-star ratings to thumbs system"""
        test_data = pd.DataFrame({
            'userId': [1, 1, 1, 1, 1, 1],
            'movieId': [1, 2, 3, 4, 5, 6],
            'rating': [1.0, 2.5, 3.0, 4.0, 4.5, 5.0],
            'timestamp': [1000000000, 1000086400, 1000172800, 1000259200, 1000345600, 1000432000]
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        
        result = loader.convert_to_thumbs_ratings(test_data)
        
        # Check that original data is preserved
        assert 'rating' in result.columns
        assert 'thumbs_rating' in result.columns
        assert len(result) == len(test_data)
        
        # Check thumbs rating conversions
        expected_thumbs = [-1.0, -1.0, 1.0, 1.0, 2.0, 2.0]  # Based on the conditions
        assert result['thumbs_rating'].tolist() == expected_thumbs
        
        # Verify original ratings are unchanged
        assert result['rating'].tolist() == test_data['rating'].tolist()
        
        # Check that other columns are preserved
        assert result['userId'].tolist() == test_data['userId'].tolist()
        assert result['movieId'].tolist() == test_data['movieId'].tolist()

    def test_convert_to_thumbs_ratings_edge_cases(self):
        """Test thumbs rating conversion edge cases"""
        test_data = pd.DataFrame({
            'userId': [1, 1, 1],
            'movieId': [1, 2, 3],
            'rating': [0.5, 2.5, 3.5],  # Edge cases: very low, boundary, mid-range
            'timestamp': [1000000000, 1000086400, 1000172800]
        })
        
        with patch('yaml.safe_load'), patch('builtins.open'):
            loader = MovieLensLoader("dummy_config.yaml")
        
        result = loader.convert_to_thumbs_ratings(test_data)
        
        # Check specific edge case conversions
        assert result['thumbs_rating'].iloc[0] == -1.0  # 0.5 <= 2.5 -> thumbs down
        assert result['thumbs_rating'].iloc[1] == -1.0  # 2.5 <= 2.5 -> thumbs down
        assert result['thumbs_rating'].iloc[2] == 1.0   # 3.5 <= 4.0 -> thumbs up