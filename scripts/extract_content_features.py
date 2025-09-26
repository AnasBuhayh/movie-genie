"""
Content Feature Extraction Pipeline

This script extracts and engineers features from TMDB movie data for the Movie Genie project.
It combines MovieLens link data with TMDB metadata to create a comprehensive feature set.
"""

import logging
import pandas as pd
import yaml
from pathlib import Path

from movie_genie.data.embeddings import TextEmbedder
from movie_genie.data.processors import TMDBFeatureProcessor, DataMerger


def setup_logging() -> None:
    """Configure logging for the feature extraction process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/data.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_movielens_links(dataset_size: str) -> pd.DataFrame:
    """Load MovieLens links dataset."""
    links_path = Path(f"data/raw/{dataset_size}/links.csv")
    if not links_path.exists():
        raise FileNotFoundError(f"Links file not found: {links_path}")

    links_df = pd.read_csv(links_path)
    logging.info(f"Loaded {len(links_df):,} MovieLens movie links")
    return links_df


def load_tmdb_dataset() -> pd.DataFrame:
    """Load TMDB movie dataset."""
    tmdb_path = Path("data/raw/tmdb/TMDB_movie_dataset_v11.csv")
    if not tmdb_path.exists():
        raise FileNotFoundError(f"TMDB file not found: {tmdb_path}")

    tmdb_df = pd.read_csv(tmdb_path)
    logging.info(f"Loaded {len(tmdb_df):,} TMDB movies")
    return tmdb_df


def main() -> None:
    """Main execution function."""
    setup_logging()
    logging.info("Starting content feature extraction...")

    try:
        # Load configuration
        config = load_config()
        dataset_size = config['data_sources']['movielens']['dataset_size']

        # Load datasets
        links_df = load_movielens_links(dataset_size)
        tmdb_df = load_tmdb_dataset()

        # Optional: Filter by movies with ratings
        filter_by_ratings = config.get('processing', {}).get('filter_by_ratings', False)
        if filter_by_ratings:
            ratings_path = Path(f"data/raw/{dataset_size}/ratings.csv")
            links_df = DataMerger.filter_by_ratings(links_df, ratings_path)

        # Merge datasets
        enriched_movies = DataMerger.merge_tmdb_with_links(links_df, tmdb_df)

        # Initialize processors
        embedder = TextEmbedder()
        processor = TMDBFeatureProcessor(embedder)

        # Extract features
        enriched_movies = processor.process_features(enriched_movies)

        # Save results
        output_path = Path("data/processed/content_features.parquet")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        enriched_movies.to_parquet(output_path)

        logging.info(f"Saved {len(enriched_movies):,} movies with content features to {output_path}")
        logging.info("Content feature extraction completed successfully")

    except Exception as e:
        logging.error(f"Content feature extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()