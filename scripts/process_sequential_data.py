import pandas as pd
from pathlib import Path
from movie_genie.data.loaders import MovieLensLoader

def main():
    print("Starting sequential processing...")
    
    # Read intermediate data from stage 1
    input_path = Path("data/interim/thumbs_ratings.parquet")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}. Run 'dvc repro ingest' first.")
    
    thumbs_df = pd.read_parquet(input_path)
    print(f"Loaded {len(thumbs_df):,} converted ratings")
    
    # Initialize loader for methods (but don't use its loading functions)
    loader = MovieLensLoader("configs/data.yaml")
    
    # Run sequential processing pipeline
    processed_df, analysis_results = loader.analyze_user_rating_patterns(thumbs_df)
    print(f"Analysis complete: {analysis_results['median_user_ratings']} median ratings per user")
    
    user_categories = loader.categorize_users_by_activity(processed_df)
    user_gaps = processed_df.groupby('userId')['days_since_last'].median()
    user_windows = loader.calculate_adaptive_windows(user_categories, user_gaps)
    
    sequences_df = loader.create_adaptive_sequences(processed_df, user_windows)
    
    # Save outputs
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    sequences_df.to_parquet(output_dir / "sequences_with_metadata.parquet")
    user_windows.to_parquet(output_dir / "user_windows.parquet")
    
    print(f"Saved {len(sequences_df):,} sequence interactions")
    print(f"Found {sequences_df['sequence_id'].nunique():,} sequences")

if __name__ == "__main__":
    main()