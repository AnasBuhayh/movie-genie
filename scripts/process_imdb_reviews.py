


import logging
from typing import Dict, List, Optional
from pathlib import Path
import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def load_imdb_dataset() -> pd.DataFrame:
    """Load IMDB movie dataset.

    Returns:
        DataFrame with IMDB movie data

    Raises:
        FileNotFoundError: If IMDB file doesn't exist
    """
    imdb_path = Path("data/raw/imdb-reviews/all_reviews.csv")
    if not imdb_path.exists():
        raise FileNotFoundError(f"IMDB file not found: {imdb_path}")

    imdb_df = pd.read_csv(imdb_path)
    logging.info(f"Loaded {len(imdb_df):,} IMDB reviews")

    # Convert IMDb ID format: tt0123456 -> 123456
    imdb_df['imdb_id_int'] = imdb_df['imdb_id'].str.replace('tt', '').astype(int)

    return imdb_df

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

def extract_text_features(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """Extract text features and embeddings from movie data."""
    logging.info("Processing text features...")
    
    # Get title and content
    df['combined_text'] = df['title'] + " " + df['content']

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


def merge_datasets(imdb_df: pd.DataFrame, links_df: pd.DataFrame) -> pd.DataFrame:
    """Merge IMDB and MovieLens datasets on movie IDs.

    Args:
        imdb_df: DataFrame with IMDB movie data
        links_df: DataFrame with MovieLens link data

    Returns:
        Merged DataFrame
    """
    # Merge on integer imdbId (imdb_df has imdb_id_int, links has imdbId)
    merged_df = pd.merge(imdb_df, links_df, left_on="imdb_id_int", right_on="imdbId", how="inner")
    logging.info(f"Merged datasets: {len(merged_df):,} reviews for {merged_df['movieId'].nunique():,} movies")
    return merged_df

def main():
    setup_logging()
    config = load_config()
    
    dataset_size = config['data_sources']['movielens']['dataset_size']
    force_reprocess = config['processing']['force_reprocess']
    
    imdb_df = load_imdb_dataset()
    links_df = load_movielens_links(dataset_size)
    
    merged_df = merge_datasets(imdb_df, links_df)
    
    model, tokenizer = load_embedding_model()
    
    processed_df = extract_text_features(merged_df, model, tokenizer)
    
    # Save as parquet
    output_path = Path("data/processed/imdb_reviews_with_embeddings.parquet")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    processed_df.to_parquet(output_path)
    logging.info(f"Saved {len(processed_df):,} reviews with embeddings to {output_path}")

if __name__ == "__main__":
    main()