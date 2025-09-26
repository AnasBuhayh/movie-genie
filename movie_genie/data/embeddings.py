"""
Text Embedding Utilities

This module provides functionality for generating text embeddings using
pre-trained language models for the Movie Genie recommendation system.
"""

import logging
import torch
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm


class TextEmbedder:
    """Handles text embedding generation using pre-trained language models."""

    def __init__(self, model_name: str = "google/embeddinggemma-300M"):
        """Initialize the text embedder.

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModel

            logging.info(f"Loading embedding model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode

            logging.info("Embedding model loaded successfully")

        except Exception as e:
            logging.warning(f"Could not load embedding model: {e}")
            logging.info("Continuing without embeddings...")
            self.model = None
            self.tokenizer = None

    def is_available(self) -> bool:
        """Check if the embedding model is available."""
        return self.model is not None and self.tokenizer is not None

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per batch

        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """
        if not self.is_available():
            logging.warning("Model not available, returning zero embeddings")
            return np.zeros((len(texts), 768))  # Default embedding dimension

        if not texts:
            return np.array([])

        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._process_batch(batch_texts)
            embeddings.append(batch_embeddings)

        result = np.vstack(embeddings)
        logging.info(f"Generated {result.shape[0]:,} embeddings of dimension {result.shape[1]}")
        return result

    def _process_batch(self, texts: List[str]) -> np.ndarray:
        """Process a single batch of texts.

        Args:
            texts: List of text strings for this batch

        Returns:
            Array of embeddings for this batch
        """
        # Tokenize the batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling across sequence length
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)

        return batch_embeddings.cpu().numpy()


def combine_text_fields(*fields) -> str:
    """Combine multiple text fields into a single string.

    Args:
        *fields: Variable number of text fields to combine

    Returns:
        Combined text string
    """
    clean_fields = []
    for field in fields:
        if field and str(field).strip():
            clean_fields.append(str(field).strip())

    return " ".join(clean_fields)


def preprocess_movie_text(overview: Optional[str] = None,
                         tagline: Optional[str] = None,
                         keywords: Optional[List[str]] = None) -> str:
    """Preprocess movie text fields for embedding generation.

    Args:
        overview: Movie overview/description
        tagline: Movie tagline
        keywords: List of movie keywords

    Returns:
        Combined and cleaned text string
    """
    text_parts = []

    if overview:
        text_parts.append(str(overview).strip())

    if tagline:
        text_parts.append(str(tagline).strip())

    if keywords:
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        text_parts.append(keywords_str.strip())

    return " ".join(text_parts).strip()


def preprocess_review_text(title: Optional[str] = None,
                          content: Optional[str] = None) -> str:
    """Preprocess review text fields for embedding generation.

    Args:
        title: Review title
        content: Review content/body

    Returns:
        Combined and cleaned text string
    """
    return combine_text_fields(title, content)