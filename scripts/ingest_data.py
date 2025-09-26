from movie_genie.data.loaders import MovieLensLoader
from pathlib import Path

def main():
    print("Starting data ingest...")
    
    # Use your existing class for loading and conversion
    loader = MovieLensLoader("configs/data.yaml")
    
    # Load raw ratings and convert to thumbs
    raw_df = loader.load_ratings()
    print(f"Loaded {len(raw_df):,} raw ratings")
    
    converted_df = loader.convert_to_thumbs_ratings(raw_df)
    
    # Save to intermediate location
    output_path = Path("data/interim/thumbs_ratings.parquet")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    converted_df.to_parquet(output_path)
    
    print(f"Saved {len(converted_df):,} converted ratings")
    print(f"Thumbs distribution:\n{converted_df['thumbs_rating'].value_counts().sort_index()}")

if __name__ == "__main__":
    main()