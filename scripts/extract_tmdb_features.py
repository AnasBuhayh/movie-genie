"""
TMDB Feature Extraction Script

This script extracts and engineers features from TMDB movie data for the Movie Genie project.
It combines MovieLens link data with TMDB metadata to create a comprehensive feature set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import yaml
import json
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

# Constants
TOP_LANGUAGES_COUNT = 15
SEASON_MAPPING = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'fall': [9, 10, 11]
}

def load_embedding_model():
    """Load EmbeddingGemma model optimized for semantic similarity tasks.
    
    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading fails
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Using the embedding-optimized model instead of general language model
        model_name = "google/embeddinggemma-300M"  # Specialized for embeddings
        logging.info(f"Loading embedding model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode for inference
        
        logging.info("EmbeddingGemma model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logging.warning(f"Could not load embedding model: {e}")
        logging.info("Continuing without embeddings...")
        return None, None

def setup_logging() -> None:
    """Configure logging for the feature extraction process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config() -> Dict:
    """Load configuration from YAML file.

    Returns:
        Dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path("configs/data.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_movielens_links(dataset_size: str) -> pd.DataFrame:
    """Load MovieLens links dataset.

    Args:
        dataset_size: Size of the MovieLens dataset (e.g., "ml-25m")

    Returns:
        DataFrame with MovieLens link data

    Raises:
        FileNotFoundError: If links file doesn't exist
    """
    links_path = Path(f"data/raw/{dataset_size}/links.csv")
    if not links_path.exists():
        raise FileNotFoundError(f"Links file not found: {links_path}")

    links_df = pd.read_csv(links_path)
    logging.info(f"Loaded {len(links_df):,} MovieLens movie links")
    return links_df

def load_tmdb_dataset() -> pd.DataFrame:
    """Load TMDB movie dataset.

    Returns:
        DataFrame with TMDB movie data

    Raises:
        FileNotFoundError: If TMDB file doesn't exist
    """
    tmdb_path = Path("data/raw/tmdb/TMDB_movie_dataset_v11.csv")
    if not tmdb_path.exists():
        raise FileNotFoundError(f"TMDB file not found: {tmdb_path}")

    tmdb_df = pd.read_csv(tmdb_path)
    logging.info(f"Loaded {len(tmdb_df):,} TMDB movies")
    return tmdb_df

def merge_datasets(links_df: pd.DataFrame, tmdb_df: pd.DataFrame) -> pd.DataFrame:
    """Merge MovieLens links with TMDB data.

    Args:
        links_df: MovieLens links dataframe
        tmdb_df: TMDB movies dataframe

    Returns:
        Merged dataframe with both datasets
    """
    enriched_movies = links_df.merge(
        tmdb_df,
        left_on='tmdbId',
        right_on='id',
        how='left'
    )

    logging.info(f"Merged datasets: {len(enriched_movies):,} movies")

    # Check TMDB coverage
    has_tmdb_data = enriched_movies['title'].notna()
    coverage_pct = has_tmdb_data.mean() * 100
    logging.info(f"TMDB coverage: {has_tmdb_data.sum():,} movies ({coverage_pct:.1f}%)")

    return enriched_movies

def extract_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean numerical features from TMDB data.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with numerical features added
    """
    logging.info("Extracting numerical features...")

    # Budget features
    df['budget'] = df['budget'].fillna(0)
    df['has_budget'] = (df['budget'] > 0).astype(int)

    # Revenue features
    df['revenue'] = df['revenue'].fillna(0)
    df['has_revenue'] = (df['revenue'] > 0).astype(int)

    # Runtime features
    df['runtime'] = df['runtime'].fillna(0)
    df['has_runtime'] = (df['runtime'] > 0).astype(int)

    # Rating features
    df['vote_average'] = df['vote_average'].fillna(0)
    df['vote_count'] = df['vote_count'].fillna(0)

    # Popularity score
    df['popularity'] = df['popularity'].fillna(0)

    # Derived features
    df['roi'] = np.where(
        df['budget'] > 0,
        df['revenue'] / df['budget'],
        0
    )

    return df

def extract_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and encode categorical features.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with categorical features added
    """
    logging.info("Processing categorical features...")

    # Adult content flag
    df['is_adult'] = df['adult'].fillna(False).astype(int)

    return df

def extract_language_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and encode language features.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with language features added
    """
    # Handle missing language data
    df['original_language'] = df['original_language'].fillna('unknown')

    # Get language distribution
    lang_counts = df['original_language'].value_counts()
    logging.info(f"Found {len(lang_counts)} unique languages")

    top_languages = lang_counts.head(TOP_LANGUAGES_COUNT).index.tolist()
    logging.info(f"Top {TOP_LANGUAGES_COUNT} languages: {top_languages}")

    # One-hot encode top languages
    for lang in top_languages:
        col_name = f'lang_{lang}'
        df[col_name] = (df['original_language'] == lang).astype(int)

    # Create 'other' category for remaining languages
    df['lang_other'] = (~df['original_language'].isin(top_languages)).astype(int)

    return df

def get_season_from_month(month: Optional[int]) -> str:
    """Convert month number to season.

    Args:
        month: Month number (1-12) or None

    Returns:
        Season name ('winter', 'spring', 'summer', 'fall', 'unknown')
    """
    if pd.isna(month):
        return 'unknown'

    for season, months in SEASON_MAPPING.items():
        if month in months:
            return season

    return 'unknown'

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date and time-based features.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with date features added
    """
    # Parse release dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['release_decade'] = (df['release_year'] // 10) * 10
    df['release_month'] = df['release_date'].dt.month

    # Add season feature
    df['release_season'] = df['release_month'].apply(get_season_from_month)

    return df

def parse_production_companies(company_string: str) -> List[str]:
    """Parse production companies JSON string into list of company names.

    Args:
        company_string: JSON string containing company data

    Returns:
        List of company names
    """
    if pd.isna(company_string) or company_string == "[]":
        return []

    try:
        companies_list = json.loads(company_string.replace("'", '"'))
        return [company['name'] for company in companies_list]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

def extract_production_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract production company features.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with production features added
    """
    # Parse production companies
    df['company_list'] = df['production_companies'].apply(parse_production_companies)

    # Extract primary production company
    df['primary_company'] = df['company_list'].apply(
        lambda companies: companies[0] if companies else 'unknown'
    )

    # Count of production companies
    df['num_companies'] = df['company_list'].apply(len)

    # Independent vs studio system
    df['is_independent'] = (df['num_companies'] <= 1).astype(int)

    return df

def preprocess_movie_text(overview, tagline, keywords):
    """Preprocess movie text fields into a single combined string."""
    # Handle missing data
    clean_overview = overview if overview else ""
    clean_tagline = tagline if tagline else ""
    clean_keywords = ", ".join(keywords) if keywords else ""

    # Combine strategically
    combined_text = f"{clean_overview} {clean_tagline} {clean_keywords}"

    return combined_text.strip()

def extract_text_features(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """Extract text features and embeddings from movie data."""
    logging.info("Processing text features...")
    
    # Combine text fields with parsed keywords
    df['combined_text'] = df.apply(lambda row: preprocess_movie_text(
        row.get('overview', ''),
        row.get('tagline', ''),
        parse_keywords(row.get('keywords', ''))
    ), axis=1)

    # Generate embeddings if model is available
    if model is not None and tokenizer is not None:
        logging.info("Generating text embeddings...")
        embeddings = process_text_batch(df['combined_text'].tolist(), model, tokenizer)
        df['text_embedding'] = embeddings.tolist()
        logging.info(f"Generated embeddings: {embeddings.shape}")
    else:
        logging.info("No model provided, skipping embedding generation")
        df['text_embedding'] = None
    
    return df

def parse_keywords(keyword_string: str) -> List[str]:
    """Parse keywords JSON string into list of keyword names.

    Args:
        keyword_string: JSON string containing keyword data

    Returns:
        List of keyword names
    """
    if pd.isna(keyword_string) or keyword_string == "[]":
        return []

    try:
        keywords_list = json.loads(keyword_string.replace("'", '"'))
        return [keyword['name'] for keyword in keywords_list]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

def process_text_batch(texts: List[str], model, tokenizer, batch_size: int = 8) -> np.ndarray:
    """Process text in batches to generate embeddings.

    Args:
        texts: List of text strings to embed
        model: Pre-trained embedding model
        tokenizer: Model tokenizer
        batch_size: Number of texts per batch (memory management)

    Returns:
        Array of embeddings (one per input text)
    """
    if model is None or tokenizer is None:
        # Return zero vectors if no model
        return np.zeros((len(texts), 384))  # Assuming 384-dim embeddings

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing text batches"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,           # Pad to same length
            truncation=True,        # Cut off if too long
            max_length=512,         # Max tokens per text
            return_tensors="pt"     # Return PyTorch tensors
        )

        # Generate embeddings
        with torch.no_grad():  # Don't compute gradients (saves memory)
            outputs = model(**inputs)
            # Use mean pooling across all tokens
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings) if embeddings else np.array([])

def engineer_features(df: pd.DataFrame, model=None, tokenizer=None) -> pd.DataFrame:
    """Apply all feature engineering steps.

    Args:
        df: Input dataframe with raw TMDB data

    Returns:
        Dataframe with engineered features
    """
    df = extract_numerical_features(df)
    df = extract_categorical_features(df)
    df = extract_language_features(df)
    df = extract_date_features(df)
    df = extract_production_features(df)
    df = extract_text_features(df, model=model, tokenizer=tokenizer)

    return df

def main() -> None:
    """Main execution function."""
    setup_logging()
    logging.info("Starting TMDB feature extraction...")

    try:
        # Load configuration and data
        config = load_config()
        dataset_size = config['data_sources']['movielens']['dataset_size']

        links_df = load_movielens_links(dataset_size)
        tmdb_df = load_tmdb_dataset()

        # Merge datasets
        enriched_movies = merge_datasets(links_df, tmdb_df)

        model, tokenizer = load_embedding_model()

        # Apply feature engineering
        enriched_movies = engineer_features(enriched_movies, model, tokenizer)

        logging.info("Feature extraction completed successfully")

        output_path = Path("data/processed/movies_with_content_features.parquet")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        enriched_movies.to_parquet(output_path)
        
        logging.info(f"Saved {len(enriched_movies):,} movies with features")

    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()