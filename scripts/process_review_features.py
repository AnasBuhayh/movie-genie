"""
Review Feature Processing Pipeline

This script processes IMDb reviews and generates embeddings for the Movie Genie project.
It merges review data with MovieLens IDs and creates text embeddings.
"""

import logging
import pandas as pd
import yaml
from pathlib import Path

from movie_genie.data.embeddings import TextEmbedder
from movie_genie.data.processors import IMDbReviewProcessor


def setup_logging() -> None:
    """Configure logging for the review processing."""
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


def load_imdb_reviews() -> pd.DataFrame:
    """Load IMDb reviews dataset."""
    reviews_path = Path("data/raw/imdb-reviews/all_reviews.csv")
    if not reviews_path.exists():
        raise FileNotFoundError(f"IMDb reviews file not found: {reviews_path}")

    reviews_df = pd.read_csv(reviews_path)
    logging.info(f"Loaded {len(reviews_df):,} IMDb reviews")
    return reviews_df


def load_movielens_links(dataset_size: str) -> pd.DataFrame:
    """Load MovieLens links dataset."""
    links_path = Path(f"data/raw/{dataset_size}/links.csv")
    if not links_path.exists():
        raise FileNotFoundError(f"Links file not found: {links_path}")

    links_df = pd.read_csv(links_path)
    logging.info(f"Loaded {len(links_df):,} MovieLens movie links")
    return links_df


def main() -> None:
    """Main execution function."""
    setup_logging()
    logging.info("Starting review feature processing...")

    try:
        # Load configuration
        config = load_config()
        dataset_size = config['data_sources']['movielens']['dataset_size']

        # Load datasets
        reviews_df = load_imdb_reviews()
        links_df = load_movielens_links(dataset_size)

        # Initialize processor
        embedder = TextEmbedder()
        processor = IMDbReviewProcessor(embedder)

        # Process reviews
        processed_reviews = processor.process_reviews(reviews_df, links_df)

        # Save results
        output_path = Path("data/processed/review_features.parquet")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        processed_reviews.to_parquet(output_path)

        logging.info(f"Saved {len(processed_reviews):,} processed reviews to {output_path}")
        logging.info("Review feature processing completed successfully")

    except Exception as e:
        logging.error(f"Review feature processing failed: {e}")
        raise


if __name__ == "__main__":
    main()