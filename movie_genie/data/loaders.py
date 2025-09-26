import pandas as pd
import numpy as np  # Add this for the rating conversion
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

class MovieLensLoader:
    def __init__(self, config_path: str = "configs/data.yaml"):
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Build paths from config
        dataset_size = self.config['data_sources']['movielens']['dataset_size']
        self.data_path = Path(f"data/raw/{dataset_size}")
        self.cache_dir = self.config['data_sources']['movielens']['cache_dir']
        
        self.ratings_cache = None
        self.movies_cache = None

    def load_ratings(self) -> pd.DataFrame:
        if self.ratings_cache is not None:
            return self.ratings_cache
            
        ratings_file = self.data_path / "ratings.csv"
        if not ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
        
        # Load with optimized data types for memory efficiency
        dtype_map = {
            'userId': 'int32',      # Saves memory vs default int64
            'movieId': 'int32', 
            'rating': 'float32',    # Sufficient precision for ratings
            'timestamp': 'int64'    # Unix timestamps need full range
        }
        
        df = pd.read_csv(ratings_file, dtype=dtype_map) # type: ignore
        
        self.ratings_cache = df
        return df
    
    def calculate_adaptive_windows(self, user_categories: pd.DataFrame, user_gaps: pd.Series) -> pd.DataFrame:
        # Merge categories with their actual gap data
        user_data = user_categories.merge(
            user_gaps.reset_index().rename(columns={'days_since_last': 'median_gap_days'}), 
            on='userId'
        )
        
        # Apply 2.5x multiplier for sequence windows
        user_data['sequence_window_days'] = user_data['median_gap_days'] * 2.5
        
        return user_data[['userId', 'user_category', 'median_gap_days', 'sequence_window_days']]
    
    def categorize_users_by_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate median rating gap for each user
        user_gaps = df.groupby('userId')['days_since_last'].median()
        
        # Create quartile-based categories
        quartiles = user_gaps.quantile([0.25, 0.5, 0.75])
        
        def assign_category(gap):
            if pd.isna(gap):  # Handle users with only one rating
                return 'single_rating'
            elif gap <= quartiles[0.25]:
                return 'heavy'  # Q1 - most frequent raters
            elif gap <= quartiles[0.5]:
                return 'moderate_high'  # Q2
            elif gap <= quartiles[0.75]:
                return 'moderate_low'   # Q3
            else:
                return 'casual'  # Q4 - least frequent raters
        
        user_categories = user_gaps.apply(assign_category).reset_index()
        user_categories.columns = ['userId', 'user_category']
        
        return user_categories
    
    from typing import Tuple

    def analyze_user_rating_patterns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Analyze actual user behavior to inform sequential modeling decisions
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values(['userId', 'datetime'])
        
        # Calculate gaps between consecutive ratings per user
        df['days_since_last'] = df.groupby('userId')['datetime'].diff().dt.days
        
        user_stats = df.groupby('userId').agg({
            'movieId': 'count',                    # Total ratings per user
            'days_since_last': ['mean', 'median', 'std'],  # Rating frequency patterns
            'datetime': lambda x: (x.max() - x.min()).days  # Active period length
        }).round(2)
        
        # Flatten column names
        user_stats.columns = ['total_ratings', 'avg_gap_days', 'median_gap_days', 'std_gap_days', 'active_period_days']
        
        return df, {
            'median_user_ratings': user_stats['total_ratings'].median(),
            'median_rating_gap_days': user_stats['median_gap_days'].median(),
            'heavy_users_threshold': user_stats['total_ratings'].quantile(0.8),
            'recommended_sequence_window': int(user_stats['median_gap_days'].median() * 3)
        }
    
    def create_adaptive_sequences(self, df: pd.DataFrame, user_windows: pd.DataFrame) -> pd.DataFrame:
        # Merge ratings with user-specific sequence windows
        df_with_windows = df.merge(user_windows[['userId', 'sequence_window_days', 'user_category']], on='userId')
        
        # Sort by user and time for proper sequence detection
        df_with_windows = df_with_windows.sort_values(['userId', 'datetime'])
        
        # Calculate if this rating starts a new sequence
        df_with_windows['days_since_last'] = df_with_windows.groupby('userId')['datetime'].diff().dt.days
        df_with_windows['new_sequence'] = (
            df_with_windows['days_since_last'] > df_with_windows['sequence_window_days']
        ) | df_with_windows['days_since_last'].isna()

        # Generate sequence IDs based on the break points
        df_with_windows['sequence_num'] = df_with_windows.groupby('userId')['new_sequence'].cumsum()
        df_with_windows['sequence_id'] = (
            df_with_windows['userId'].astype(str) + '_seq_' + 
            df_with_windows['sequence_num'].astype(str)
        )
        
        return df_with_windows[['userId', 'movieId', 'rating', 'thumbs_rating', 'datetime', 'sequence_id', 'user_category']]
    
    def save_processed_sequences(self, df: pd.DataFrame, user_windows: pd.DataFrame, cache_dir: str = "data/processed"):
        
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True, parents=True)
        
        # Save the processed sequences and user data
        df.to_parquet(cache_path / "sequences_with_metadata.parquet")
        user_windows.to_parquet(cache_path / "user_windows.parquet")
        
        print(f"Saved processed data to {cache_path}")

    def convert_to_thumbs_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert MovieLens 5-star ratings to Netflix-style thumbs system
        """
        df = df.copy()
        
        # Create thumbs rating based on original rating
        conditions = [
            df['rating'] <= 2.5,  # thumbs down
            df['rating'] <= 4.0,  # thumbs up  
            df['rating'] <= 5.0   # two thumbs up
        ]
        
        choices = [-1.0, 1.0, 2.0]
        
        df['thumbs_rating'] = np.select(conditions, choices)
        
        # Keep original rating for analysis, use thumbs for modeling
        return df

    def load_or_process_sequences(self, force_reprocess: bool = False, cache_dir: str = "data/processed") -> tuple:
        from pathlib import Path
        
        cache_path = Path(cache_dir)
        sequences_file = cache_path / "sequences_with_metadata.parquet"
        user_windows_file = cache_path / "user_windows.parquet"
        
        # Check if cache exists and we're not forcing reprocessing
        if not force_reprocess and sequences_file.exists() and user_windows_file.exists():
            print("Loading cached processed data...")
            sequences_df = pd.read_parquet(sequences_file)
            user_windows_df = pd.read_parquet(user_windows_file)
            return sequences_df, user_windows_df

        # Cache doesn't exist or we're forcing reprocess - do the full pipeline
        print("Processing data from scratch...")
        
        # Step 1: Load raw ratings
        raw_df = self.load_ratings()

        # Step 1.5: Convert to thumbs rating system
        raw_df = self.convert_to_thumbs_ratings(raw_df)
        
        # Step 2: Analyze user behavior patterns and get processed DataFrame
        processed_df, analysis_results = self.analyze_user_rating_patterns(raw_df)
        print(f"Analysis complete: {analysis_results['median_user_ratings']} median ratings per user")
        
        # Step 3: Categorize users by activity level
        user_categories = self.categorize_users_by_activity(processed_df)
        
        # Step 4: Calculate adaptive windows for each user category
        user_gaps = processed_df.groupby('userId')['days_since_last'].median()
        user_windows = self.calculate_adaptive_windows(user_categories, user_gaps)
        
        # Step 5: Create sequences using adaptive windows
        sequences_df = self.create_adaptive_sequences(processed_df, user_windows)
        
        # Save to cache for next time
        self.save_processed_sequences(sequences_df, user_windows, cache_dir)
        
        return sequences_df, user_windows