from movie_genie.data.loaders import MovieLensLoader
from pathlib import Path
import yaml
import shutil

def main():
    print("Starting data ingest...")

    # Load config
    with open("configs/data.yaml", 'r') as f:
        config = yaml.safe_load(f)

    dataset_size = config['data_sources']['movielens']['dataset_size']
    source_dir = Path(f"data/raw/{dataset_size}")

    # Verify source data exists
    if not source_dir.exists():
        raise FileNotFoundError(f"MovieLens data not found at {source_dir}")

    if not (source_dir / "ratings.csv").exists():
        raise FileNotFoundError(f"ratings.csv not found in {source_dir}")

    if not (source_dir / "movies.csv").exists():
        raise FileNotFoundError(f"movies.csv not found in {source_dir}")

    # Copy to DVC output location
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy ratings and movies to expected output paths
    shutil.copy(source_dir / "ratings.csv", output_dir / "ratings.csv")
    shutil.copy(source_dir / "movies.csv", output_dir / "movies.csv")

    print(f"Copied ratings.csv to {output_dir / 'ratings.csv'}")
    print(f"Copied movies.csv to {output_dir / 'movies.csv'}")

    # Load and verify
    loader = MovieLensLoader("configs/data.yaml")
    raw_df = loader.load_ratings()
    print(f"Loaded {len(raw_df):,} raw ratings")

    converted_df = loader.convert_to_thumbs_ratings(raw_df)
    print(f"Saved {len(converted_df):,} converted ratings")
    print(f"Thumbs distribution:\n{converted_df['thumbs_rating'].value_counts().sort_index()}")

if __name__ == "__main__":
    main()
