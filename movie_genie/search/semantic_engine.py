"""
Semantic Search Engine for Movie Discovery

This module creates the core infrastructure for semantic search by leveraging
the existing EmbeddingGemma text embeddings computed during content feature engineering.

The key insight: your movie embeddings already capture sophisticated semantic
relationships between films. Semantic search extends this by mapping user queries
into the same mathematical space, enabling natural language movie discovery.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import yaml
import re

# Import our existing text embedding utilities for consistency
import sys
sys.path.append('.')
from movie_genie.data.embeddings import TextEmbedder

# Import rerankers
from movie_genie.search.rerankers import SearchReranker, BERT4RecReranker, TwoTowerReranker, NoOpReranker

class MovieEmbeddingLoader:
    """
    Loads and prepares existing movie embeddings for semantic search.
    
    This class handles the practical challenge of extracting the text embeddings
    you computed during Stage 4 processing and organizing them for fast similarity
    search operations. Understanding this component helps you see how semantic search
    builds directly on your existing content processing work.
    """
    
    def __init__(self, movies_with_features_path: str):
        """
        Initialize the loader with your processed movie data.
        
        The initialization process needs to handle the fact that your embeddings
        are stored within a larger dataframe that contains many other features.
        We need to extract just the text embeddings while maintaining the connection
        to movie metadata for result presentation.
        
        Args:
            movies_with_features_path: Path to your Stage 4 processed movie data
        """
        self.movies_path = Path(movies_with_features_path)
        self.movies_df = None
        self.embeddings_matrix = None
        self.movie_metadata = None
        
        # Load and validate the movie data
        self._load_and_validate_data()
    
    def _load_and_validate_data(self) -> None:
        """
        Load movie data and extract embeddings with careful validation.
        
        This method handles the practical reality that your processed data might
        have inconsistencies - some movies might have missing embeddings, others
        might have embeddings stored in different formats. We need robust handling
        that ensures we only work with valid, complete embedding data.
        """
        logging.info(f"Loading movie data from {self.movies_path}")
        
        # Load your processed movie data
        self.movies_df = pd.read_parquet(self.movies_path)
        logging.info(f"Loaded {len(self.movies_df)} total movie records")
        
        # Extract and validate text embeddings
        valid_embeddings = []
        valid_metadata = []
        
        for idx, row in self.movies_df.iterrows():
            embedding = self._extract_embedding_safely(row)
            
            if embedding is not None:
                valid_embeddings.append(embedding)
                valid_metadata.append(self._extract_movie_metadata(row))
        
        if len(valid_embeddings) == 0:
            raise ValueError("No valid text embeddings found in movie data")
        
        # Convert to numpy arrays for efficient computation
        self.embeddings_matrix = np.array(valid_embeddings, dtype=np.float32)
        self.movie_metadata = valid_metadata
        
        logging.info(f"Extracted {len(valid_embeddings)} valid movie embeddings")
        logging.info(f"Embedding dimension: {self.embeddings_matrix.shape[1]}")
    
    def _extract_embedding_safely(self, movie_row) -> Optional[np.ndarray]:
        """
        Safely extract text embedding from a movie row, handling various storage formats.

        This method demonstrates defensive programming for data processing. Your
        embeddings might be stored as lists, numpy arrays, or other formats depending
        on how the parquet serialization handled them. We need to handle all these
        cases gracefully.

        Supports embeddings of any dimension (384 for all-MiniLM, 768 for EmbeddingGemma, etc.)
        """
        text_embedding = movie_row.get('text_embedding')

        if text_embedding is None:
            return None

        try:
            # Handle different possible storage formats
            if isinstance(text_embedding, list):
                if len(text_embedding) > 0:  # Accept any valid dimension
                    return np.array(text_embedding, dtype=np.float32)
            elif isinstance(text_embedding, np.ndarray):
                if len(text_embedding.shape) == 1 and text_embedding.shape[0] > 0:
                    return text_embedding.astype(np.float32)
            else:
                # Try to convert other formats
                converted = np.array(text_embedding)
                if len(converted.shape) == 1 and converted.shape[0] > 0:
                    return converted.astype(np.float32)

        except (ValueError, TypeError) as e:
            # Log the error but continue processing other movies
            logging.debug(f"Failed to extract embedding for movie {movie_row.get('title', 'Unknown')}: {e}")

        return None
    
    def _extract_movie_metadata(self, movie_row) -> Dict[str, Any]:
        """
        Extract essential movie metadata for search result presentation.

        When users see search results, they need enough information to understand
        what each movie is about and decide if it matches their intent. This method
        selects the most important metadata fields for result presentation.
        """
        return {
            'movieId': movie_row.get('movieId'),
            'title': movie_row.get('title', 'Unknown Title'),
            'overview': movie_row.get('overview', ''),
            'genres': movie_row.get('genres', []),
            'release_date': movie_row.get('release_date', ''),
            'vote_average': movie_row.get('vote_average', 0.0),
            'vote_count': movie_row.get('vote_count', 0),
            'poster_path': movie_row.get('poster_path', None),
            'runtime': movie_row.get('runtime', None)
        }
    
    def get_embeddings_and_metadata(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Return the processed embeddings and metadata for search engine use.
        
        This method provides the core data structures that the semantic search
        engine needs: a matrix of embeddings for similarity computation and
        corresponding metadata for result presentation.
        """
        return self.embeddings_matrix, self.movie_metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about the loaded embedding data for diagnostics."""
        if self.embeddings_matrix is None:
            return {'status': 'not_loaded'}
        
        return {
            'total_movies': len(self.movie_metadata),
            'embedding_dimension': self.embeddings_matrix.shape[1],
            'memory_usage_mb': self.embeddings_matrix.nbytes / (1024 * 1024),
            'sample_titles': [movie['title'] for movie in self.movie_metadata[:5]]
        }
    
class QueryEncoder:
    """
    Encodes natural language queries into semantic vectors for movie search.
    
    This encoder transforms user queries into dense vector representations that
    can be compared against movie embeddings using similarity metrics. The encoding
    process handles text normalization, abbreviation expansion, and vector
    generation using pre-trained transformer models.
    
    All behavior parameters are loaded from configuration files, enabling
    experimentation with different models and preprocessing strategies without
    code changes. This design separates algorithmic logic from parameter choices.
    """
    
    def __init__(self, config_path: str = "configs/semantic_search.yaml"):
        """
        Initialize the query encoder using parameters from configuration.
        
        The configuration-driven approach enables systematic experimentation
        with different sentence transformer models, preprocessing options, and
        caching strategies. This flexibility supports the iterative improvement
        process essential for optimizing search quality.
        
        Args:
            config_path: Path to YAML configuration file defining encoder behavior
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Extract parameters from simplified configuration
        self.model_name = self.config['model_name']
        self.cache_limit = self.config['cache_size']
        self.normalize_vectors = self.config['normalize_vectors']

        # Extract preprocessing parameters
        self.abbreviations = self.config.get('abbreviations', {})
        
        # Initialize the text embedder using our existing pipeline
        logging.info(f"Loading embedding model: {self.model_name}")
        self.encoder = TextEmbedder(model_name=self.model_name)

        if not self.encoder.is_available():
            raise RuntimeError(f"Failed to load embedding model: {self.model_name}")

        # Determine embedding dimension by encoding a test query
        test_embedding = self.encoder.embed_texts(["test query"])
        self.encoding_dimension = test_embedding.shape[1]

        # Initialize query cache for performance optimization
        self.query_cache = {}

        logging.info(f"Query encoder ready: {self.encoding_dimension}-dimensional vectors")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration parameters from YAML file."""
        try:
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            
            # Validate required configuration keys
            required_keys = ['model_name', 'normalize_vectors', 'cache_size']
            missing_keys = [key for key in required_keys if key not in config]

            if missing_keys:
                raise ValueError(f"Configuration missing required keys: {missing_keys}")

            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {self.config_path}: {e}")
    
    def encode(self, query_text: str) -> np.ndarray:
        """
        Encode a natural language query into a semantic vector.
        
        This method performs the core transformation that enables semantic search:
        converting arbitrary user input into mathematical representations that can
        be compared against movie embeddings using similarity metrics.
        
        The encoding process applies configured preprocessing steps, uses the
        specified transformer model, and optionally normalizes the output vector
        for cosine similarity computation.
        
        Args:
            query_text: Natural language query from user
            
        Returns:
            Dense vector representation in semantic embedding space
        """
        # Apply text preprocessing based on configuration
        processed_query = self._preprocess_text(query_text)
        
        # Check cache for previously encoded queries
        if processed_query in self.query_cache:
            return self.query_cache[processed_query]
        
        # Generate embedding using our text embedder
        embedding = self.encoder.embed_texts([processed_query])
        query_vector = embedding[0].astype(np.float32)
        
        # Apply vector normalization if configured
        if self.normalize_vectors:
            query_vector = self._normalize_vector(query_vector)
        
        # Cache result for future queries
        self._update_cache(processed_query, query_vector)
        
        return query_vector
    
    def encode_batch(self, queries: List[str]) -> np.ndarray:
        """
        Encode multiple queries efficiently in a single batch.
        
        Batch processing provides significant performance improvements when
        encoding many queries simultaneously, such as during evaluation or
        when processing historical search logs for analysis.
        
        Args:
            queries: List of natural language queries
            
        Returns:
            Matrix where each row represents one encoded query vector
        """
        # Preprocess all queries consistently
        processed_queries = [self._preprocess_text(q) for q in queries]
        
        # Separate cached and uncached queries for efficiency
        cached_results = {}
        uncached_queries = []
        uncached_indices = []
        
        for i, query in enumerate(processed_queries):
            if query in self.query_cache:
                cached_results[i] = self.query_cache[query]
            else:
                uncached_queries.append(query)
                uncached_indices.append(i)
        
        # Initialize result matrix
        result_matrix = np.zeros((len(queries), self.encoding_dimension), dtype=np.float32)
        
        # Fill in cached results
        for index, cached_vector in cached_results.items():
            result_matrix[index] = cached_vector
        
        # Batch encode uncached queries
        if uncached_queries:
            batch_embeddings = self.encoder.embed_texts(uncached_queries)
            
            for i, (result_index, query) in enumerate(zip(uncached_indices, uncached_queries)):
                vector = batch_embeddings[i].astype(np.float32)
                
                if self.normalize_vectors:
                    vector = self._normalize_vector(vector)
                
                result_matrix[result_index] = vector
                self._update_cache(query, vector)
        
        return result_matrix
    
    def _preprocess_text(self, text: str) -> str:
        """Apply simple text preprocessing transformations to query text."""
        processed = text.strip().lower()

        # Normalize whitespace
        processed = ' '.join(processed.split())

        # Expand abbreviations
        processed = self._expand_abbreviations(processed)

        return processed
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand movie-domain abbreviations using configured mappings."""
        expanded = text
        for abbreviation, expansion in self.abbreviations.items():
            expanded = expanded.replace(abbreviation, expansion)
        return expanded
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine similarity computation."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        else:
            logging.warning("Encountered zero vector during normalization")
            return vector
    
    def _update_cache(self, query: str, vector: np.ndarray) -> None:
        """Update query cache with size management to prevent memory growth."""
        # Simple LRU eviction when cache exceeds limit
        if len(self.query_cache) >= self.cache_limit:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[query] = vector
    
    def get_info(self) -> Dict[str, Any]:
        """Return encoder information for debugging and monitoring."""
        return {
            'model_name': self.model_name,
            'encoding_dimension': self.encoding_dimension,
            'normalize_vectors': self.normalize_vectors,
            'cached_queries': len(self.query_cache),
            'cache_limit': self.cache_limit,
            'config_file': str(self.config_path)
        }
    
    def clear_cache(self) -> None:
        """Clear the query cache to free memory or reset for testing."""
        self.query_cache.clear()
        logging.info("Query encoding cache cleared")

from .rerankers import BERT4RecReranker, TwoTowerReranker, NoOpReranker

class SemanticSearchEngine:
    def __init__(self, config_path: str = "configs/semantic_search.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Determine project root for resolving relative paths
        self.project_root = self._find_project_root()

        # Initialize core components
        self.query_encoder = QueryEncoder(config_path)

        # Resolve movies_path relative to project root
        movies_path = self.config['movies_path']
        if not Path(movies_path).is_absolute():
            movies_path = str(self.project_root / movies_path)

        self.movie_loader = MovieEmbeddingLoader(movies_path)
        self.movie_embeddings, self.movie_metadata = self.movie_loader.get_embeddings_and_metadata()
        
        # Initialize reranker based on configuration
        self.reranker = self._initialize_reranker()
        
        # Pre-normalize embeddings
        self.movie_embeddings = self.movie_embeddings / np.linalg.norm(
            self.movie_embeddings, axis=1, keepdims=True
        )

    def _find_project_root(self) -> Path:
        """Find the project root directory by looking for pyproject.toml or dvc.yaml."""
        current_path = Path(__file__).resolve()

        # Search upward for project markers
        for parent in current_path.parents:
            if (parent / 'pyproject.toml').exists() or (parent / 'dvc.yaml').exists():
                return parent

        # Fallback to current working directory
        return Path.cwd()
        
    def _initialize_reranker(self) -> SearchReranker:
        """Initialize reranker based on configuration."""
        rerank_config = self.config.get('reranker', {})
        
        if not rerank_config.get('enabled', False):
            return NoOpReranker()
            
        reranker_type = rerank_config.get('type', 'none')
        
        if reranker_type == 'bert4rec':
            # Resolve paths relative to project root
            model_path = rerank_config['model_path']
            if not Path(model_path).is_absolute():
                model_path = str(self.project_root / model_path)

            data_artifacts_path = rerank_config['data_artifacts_path']
            if not Path(data_artifacts_path).is_absolute():
                data_artifacts_path = str(self.project_root / data_artifacts_path)

            return BERT4RecReranker(
                model_path=model_path,
                data_artifacts_path=data_artifacts_path
            )
        elif reranker_type == 'two_tower':
            # Resolve path relative to project root
            model_path = rerank_config['model_path']
            if not Path(model_path).is_absolute():
                model_path = str(self.project_root / model_path)
            return TwoTowerReranker(model_path)
        else:
            return NoOpReranker()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration parameters from YAML file."""
        try:
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)

            # Validate required configuration keys
            required_keys = ['model_name', 'movies_path', 'default_results']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ValueError(f"Missing required configuration keys: {missing_keys}")

            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

    def search(self, query: str, k: int = None,
               user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search with optional personalized reranking."""
        if k is None:
            k = self.config.get('default_results', 20)
        
        # Get semantic search results (more candidates for reranking)
        candidate_k = min(k * 3, 100)  # Get 3x candidates for reranking
        query_vector = self.query_encoder.encode(query)
        similarities = np.dot(self.movie_embeddings, query_vector)
        top_indices = np.argsort(similarities)[-candidate_k:][::-1]
        
        # Format initial results
        results = []
        for rank, movie_idx in enumerate(top_indices):
            movie_info = self.movie_metadata[movie_idx]
            results.append({
                'movieId': movie_info['movieId'],
                'title': movie_info['title'],
                'overview': movie_info['overview'],
                'genres': movie_info.get('genres', []),
                'similarity_score': float(similarities[movie_idx]),
                'rank': rank + 1,
                # Include all metadata fields for frontend display
                'vote_average': movie_info.get('vote_average'),
                'vote_count': movie_info.get('vote_count'),
                'release_date': movie_info.get('release_date'),
                'poster_path': movie_info.get('poster_path'),
                'runtime': movie_info.get('runtime')
            })
        
        # Apply reranking if user context is provided
        reranked_results = self.reranker.rerank(results, user_context)
        
        # Return top-k after reranking
        return reranked_results[:k]
    
    def batch_search(self, queries: List[str], k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search multiple queries efficiently in batch.
        
        Args:
            queries: List of natural language queries
            k: Number of results per query
            
        Returns:
            Dictionary mapping queries to their search results
        """
        if k is None:
            k = self.config.get('default_results', 20)
        
        # Batch encode all queries
        query_vectors = self.query_encoder.encode_batch(queries)
        
        # Compute similarities for all queries at once
        similarities_matrix = np.dot(query_vectors, self.movie_embeddings.T)
        
        # Process results for each query
        batch_results = {}
        for i, query in enumerate(queries):
            similarities = similarities_matrix[i]
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for rank, movie_idx in enumerate(top_indices):
                movie_info = self.movie_metadata[movie_idx]
                results.append({
                    'movieId': movie_info['movieId'],
                    'title': movie_info['title'],
                    'overview': movie_info['overview'][:150] + '...' if len(movie_info['overview']) > 150 else movie_info['overview'],
                    'similarity_score': float(similarities[movie_idx]),
                    'rank': rank + 1
                })
            
            batch_results[query] = results
        
        return batch_results
    
    def get_similar_movies(self, movie_title: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Find movies similar to a specific movie by title.
        
        Args:
            movie_title: Title of reference movie
            k: Number of similar movies to return
            
        Returns:
            List of similar movies
        """
        # Find the reference movie
        movie_idx = None
        for idx, movie in enumerate(self.movie_metadata):
            if movie_title.lower() in movie['title'].lower():
                movie_idx = idx
                break
        
        if movie_idx is None:
            return []
        
        # Get embedding for reference movie
        movie_vector = self.movie_embeddings[movie_idx]
        
        # Compute similarities with all other movies
        similarities = np.dot(self.movie_embeddings, movie_vector)
        
        # Get top-k similar movies (excluding the reference movie itself)
        top_indices = np.argsort(similarities)[-(k+1):][::-1]
        top_indices = [idx for idx in top_indices if idx != movie_idx][:k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices):
            movie_info = self.movie_metadata[idx]
            results.append({
                'movieId': movie_info['movieId'],
                'title': movie_info['title'],
                'overview': movie_info['overview'][:200] + '...' if len(movie_info['overview']) > 200 else movie_info['overview'],
                'similarity_score': float(similarities[idx]),
                'rank': rank + 1
            })
        
        return results
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Return statistics about the search engine."""
        return {
            'total_movies': len(self.movie_metadata),
            'embedding_dimension': self.movie_embeddings.shape[1],
            'query_encoder_model': self.query_encoder.model_name,
            'config_file': str(self.config_path),
            'cached_queries': len(self.query_encoder.query_cache)
        }