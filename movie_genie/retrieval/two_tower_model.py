"""
Two-Tower Model for Movie Recommendation

This module implements a neural collaborative filtering approach using separate
user and item towers that learn to encode preferences and content into a shared
embedding space for fast similarity-based retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any

class UserTower(nn.Module):
    """
    Neural network that encodes user characteristics and behavior into embeddings.
    
    The user tower processes multiple types of user information:
    - Historical rating patterns and preferences
    - Demographic information (if available)
    - Behavioral features like rating frequency and genre preferences
    
    The goal is to create a dense representation that captures what types of
    movies this user tends to enjoy, allowing for fast similarity calculations
    against pre-computed movie embeddings.
    """
    
    def __init__(self, 
                 num_users: int,
                 user_embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64],
                 output_dim: int = 128,
                 dropout_rate: float = 0.1):
        """
        Initialize the user tower architecture.
        
        Args:
            num_users: Total number of unique users in the dataset
            user_embedding_dim: Dimension for learned user ID embeddings
            hidden_dims: List of hidden layer dimensions for the neural network
            output_dim: Final embedding dimension (must match item tower)
            dropout_rate: Dropout probability for regularization
        """
        super(UserTower, self).__init__()
        
        # User ID embedding layer - learns a unique representation for each user
        # This captures user-specific preferences that might not be evident
        # from their rating history alone
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # Calculate input dimension for the first fully connected layer
        # This will include the user embedding plus any additional user features
        input_dim = user_embedding_dim
        
        # Build the neural network layers that transform user features
        # into the final embedding representation
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),  # Non-linear activation for learning complex patterns
                nn.Dropout(dropout_rate)  # Regularization to prevent overfitting
            ])
            prev_dim = hidden_dim
        
        # Final layer that produces the output embedding
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier uniform initialization
        # This helps with training stability and convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)
    
    def forward(self, user_ids: torch.Tensor, _user_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the user tower.
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            user_features: Optional additional user features [batch_size, feature_dim]
            
        Returns:
            User embeddings [batch_size, output_dim]
        """
        # Get the learned embedding for each user ID
        user_emb = self.user_embedding(user_ids)
        
        # For now, we'll just use the user ID embedding
        # Later, you can extend this to include additional user features
        # like demographic information or computed preference features
        input_features = user_emb
        
        # Pass through the neural network to get final embeddings
        embeddings = self.network(input_features)
        
        # Normalize embeddings to unit length for cosine similarity calculations
        # This ensures that similarity scores are bounded and comparable
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
class ItemTower(nn.Module):
    """
    Neural network that encodes movie characteristics into embeddings.
    
    The item tower processes your rich TMDB content features including:
    - Numerical features like budget, revenue, runtime
    - Categorical features like language, production companies
    - Text embeddings from movie descriptions and reviews
    - Temporal features like release year and season
    
    The goal is to create embeddings that capture both explicit content
    characteristics and latent factors that influence user preferences.
    """
    
    def __init__(self,
                 num_movies: int,
                 content_feature_dim: int,
                 movie_embedding_dim: int = 64,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 128,
                 dropout_rate: float = 0.1):
        """
        Initialize the item tower architecture.
        
        Args:
            num_movies: Total number of unique movies in the dataset
            content_feature_dim: Dimension of your processed content features
            movie_embedding_dim: Dimension for learned movie ID embeddings
            hidden_dims: List of hidden layer dimensions
            output_dim: Final embedding dimension (must match user tower)
            dropout_rate: Dropout probability for regularization
        """
        super(ItemTower, self).__init__()
        
        # Movie ID embedding - learns unique representations beyond content features
        # This captures latent factors that might not be evident in explicit
        # content features but influence user preferences
        self.movie_embedding = nn.Embedding(num_movies, movie_embedding_dim)
        
        # The input dimension combines movie ID embeddings with content features
        # Your content features include TMDB metadata, text embeddings, and
        # derived features from your Stage 4 feature engineering
        input_dim = movie_embedding_dim + content_feature_dim
        
        # Build the neural network that transforms movie features into embeddings
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final layer produces embeddings in the same space as user embeddings
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)
    
    def forward(self, movie_ids: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the item tower.
        
        Args:
            movie_ids: Tensor of movie IDs [batch_size]
            content_features: Movie content features [batch_size, content_feature_dim]
            
        Returns:
            Movie embeddings [batch_size, output_dim]
        """
        # Get learned movie ID embeddings
        movie_emb = self.movie_embedding(movie_ids)
        
        # Combine movie ID embeddings with your rich content features
        # This allows the model to use both learned latent factors and
        # explicit content characteristics when creating representations
        combined_features = torch.cat([movie_emb, content_features], dim=1)
        
        # Transform through the neural network
        embeddings = self.network(combined_features)
        
        # Normalize for cosine similarity calculations
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
class TwoTowerModel(nn.Module):
    """
    Complete two-tower model that combines user and item towers for recommendation.
    
    This model learns to predict user-item compatibility by encoding users and movies
    into a shared embedding space where similarity calculations reveal preferences.
    The training process uses positive and negative examples to teach the towers
    to create compatible embeddings for preferred items and incompatible embeddings
    for items users would not enjoy.
    """
    
    def __init__(self,
                 num_users: int,
                 num_movies: int,
                 content_feature_dim: int,
                 embedding_dim: int = 128,
                 user_hidden_dims: List[int] = [128, 64],
                 item_hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1):
        """
        Initialize the complete two-tower architecture.
        
        Args:
            num_users: Total number of unique users in your dataset
            num_movies: Total number of unique movies in your dataset
            content_feature_dim: Dimension of your processed movie content features
            embedding_dim: Final embedding dimension for both towers (must match)
            user_hidden_dims: Hidden layer dimensions for user tower
            item_hidden_dims: Hidden layer dimensions for item tower
            dropout_rate: Dropout probability for regularization
        """
        super(TwoTowerModel, self).__init__()
        
        # Initialize the user tower that processes user information
        self.user_tower = UserTower(
            num_users=num_users,
            hidden_dims=user_hidden_dims,
            output_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
        # Initialize the item tower that processes movie content features
        self.item_tower = ItemTower(
            num_movies=num_movies,
            content_feature_dim=content_feature_dim,
            hidden_dims=item_hidden_dims,
            output_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
        # Store dimensions for later use in data processing
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_movies = num_movies
    
    def forward(self, 
                user_ids: torch.Tensor,
                movie_ids: torch.Tensor,
                movie_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes compatibility scores between users and movies.
        
        This method processes a batch of user-movie pairs and returns similarity
        scores that indicate how much each user would likely enjoy each movie.
        During training, these scores get compared to actual ratings to compute
        the loss function that drives the learning process.
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            movie_ids: Tensor of movie IDs [batch_size]
            movie_features: Movie content features [batch_size, content_feature_dim]
            
        Returns:
            Compatibility scores [batch_size] indicating predicted user preference
        """
        # Generate user embeddings using the user tower
        # These embeddings capture learned preference patterns for each user
        user_embeddings = self.user_tower(user_ids)
        
        # Generate movie embeddings using the item tower and content features
        # These embeddings combine learned movie characteristics with your
        # rich TMDB content features from Stage 4
        movie_embeddings = self.item_tower(movie_ids, movie_features)
        
        # Compute compatibility scores using dot product
        # Since embeddings are normalized, this gives cosine similarity
        # Higher scores indicate greater predicted compatibility
        scores = torch.sum(user_embeddings * movie_embeddings, dim=1)
        
        return scores
    
    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate embeddings for a batch of users.
        
        This method supports inference scenarios where you want to generate
        user embeddings independently, such as for caching user representations
        or performing batch similarity calculations against pre-computed movie
        embeddings.
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            
        Returns:
            User embeddings [batch_size, embedding_dim]
        """
        return self.user_tower(user_ids)
    
    def get_movie_embeddings(self,
                           movie_ids: torch.Tensor,
                           movie_features: torch.Tensor) -> torch.Tensor:
        """
        Generate embeddings for a batch of movies.
        
        This method enables pre-computing movie embeddings for fast retrieval
        during recommendation serving. Since movie content rarely changes,
        you can compute these embeddings once and store them for repeated
        similarity calculations against user embeddings.
        
        Args:
            movie_ids: Tensor of movie IDs [batch_size]
            movie_features: Movie content features [batch_size, content_feature_dim]
            
        Returns:
            Movie embeddings [batch_size, embedding_dim]
        """
        return self.item_tower(movie_ids, movie_features)
    
def compute_loss(model: TwoTowerModel,
                user_ids: torch.Tensor,
                pos_movie_ids: torch.Tensor,
                pos_movie_features: torch.Tensor,
                neg_movie_ids: torch.Tensor,
                neg_movie_features: torch.Tensor,
                margin: float = 1.0) -> torch.Tensor:
    """
    Compute the ranking loss that teaches the model to distinguish preferences.
    
    This loss function implements the mathematical framework that drives learning
    in collaborative filtering systems. The model learns by comparing positive
    examples (movies users rated highly) against negative examples (movies
    they would likely rate poorly) and adjusting embeddings to increase the
    similarity gap between these cases.
    
    Args:
        model: The two-tower model being trained
        user_ids: User IDs for the training batch [batch_size]
        pos_movie_ids: Movie IDs for positive examples [batch_size]
        pos_movie_features: Content features for positive movies [batch_size, feature_dim]
        neg_movie_ids: Movie IDs for negative examples [batch_size]
        neg_movie_features: Content features for negative movies [batch_size, feature_dim]
        margin: Minimum difference between positive and negative scores
        
    Returns:
        Loss value that drives the optimization process
    """
    # Compute similarity scores for positive user-movie pairs
    # These represent movies that users actually rated highly
    pos_scores = model(user_ids, pos_movie_ids, pos_movie_features)
    
    # Compute similarity scores for negative user-movie pairs
    # These represent movies that users would likely rate poorly
    neg_scores = model(user_ids, neg_movie_ids, neg_movie_features)
    
    # Ranking loss encourages positive scores to exceed negative scores by margin
    # This mathematical constraint teaches the model to create embeddings where
    # preferred items have higher similarity than non-preferred items
    loss = torch.relu(margin - (pos_scores - neg_scores))
    
    return loss.mean()


class TwoTowerDataLoader:
    """
    Data preparation system for two-tower model training using Netflix thumbs ratings.
    
    This class loads your processed parquet files from the DVC pipeline and
    creates training examples that leverage the clear positive/negative signals
    in your thumbs rating system. The loader handles the transformation from
    your file-based data storage into tensor format suitable for model training.
    """
    
    def __init__(self,
                 sequences_path: str = "data/processed/sequences_with_metadata.parquet",
                 movies_path: str = "data/processed/movies_with_content_features.parquet",
                 negative_sampling_ratio: int = 4,
                 min_user_interactions: int = 5):
        """
        Initialize the data loader by reading your processed parquet files.
        
        The initialization process loads both user interaction sequences and
        movie content features, then prepares them for the contrastive learning
        approach that drives two-tower model training.
        
        Args:
            sequences_path: Path to your user sequences with thumbs ratings
            movies_path: Path to your movies with content features from Stage 4
            negative_sampling_ratio: Number of negative examples per positive
            min_user_interactions: Minimum interactions required per user for training
        """
        self.negative_sampling_ratio = negative_sampling_ratio
        self.min_user_interactions = min_user_interactions
        
        logging.info("Loading processed datasets from parquet files...")
        
        # Load your user interaction sequences from the DVC pipeline
        # This contains the thumbs ratings and temporal sequence information
        self.sequences_df = pd.read_parquet(sequences_path)
        logging.info(f"Loaded {len(self.sequences_df):,} user interactions")
        
        # Load your movie content features from Stage 4 processing
        # This includes TMDB metadata, text embeddings, and derived features
        self.movies_df = pd.read_parquet(movies_path)
        logging.info(f"Loaded {len(self.movies_df):,} movies with content features")
        
        # Analyze the rating distribution to understand your data characteristics
        self._analyze_rating_distribution()
        
        # Filter users with sufficient interaction history for meaningful training
        self._filter_users_by_activity()
        
        # Create ID mappings for tensor indexing
        self._create_id_mappings()
        
        # Prepare movie content features as tensors
        self._prepare_movie_features()
        
        # Generate training examples using the thumbs rating system
        self._create_training_examples()
    
    def _analyze_rating_distribution(self):
        """
        Analyze the distribution of thumbs ratings to understand training data balance.
        
        Understanding the rating distribution helps us make informed decisions about
        sampling strategies and potential class imbalance issues that could affect
        training effectiveness.
        """
        rating_counts = self.sequences_df['thumbs_rating'].value_counts().sort_index()
        total_ratings = len(self.sequences_df)
        
        logging.info("Rating distribution analysis:")
        for rating, count in rating_counts.items():
            percentage = (count / total_ratings) * 100
            rating_type = {-1.0: "Thumbs Down", 1.0: "Thumbs Up", 2.0: "Two Thumbs Up"}
            logging.info(f"  {rating_type.get(rating, f'Rating {rating}')}: {count:,} ({percentage:.1f}%)")
        
        # Store counts for later use in sampling strategy
        self.positive_count = rating_counts.get(1.0, 0) + rating_counts.get(2.0, 0)
        self.negative_count = rating_counts.get(-1.0, 0)
        
        if self.negative_count == 0:
            logging.warning("No explicit negative ratings found. Will use implicit negatives.")
    
    def _filter_users_by_activity(self):
        """
        Filter to users with sufficient interaction history for meaningful training.
        
        Users with very few ratings provide limited signal for learning their
        preferences. By focusing on users with more substantial interaction
        histories, we improve the quality of collaborative filtering signals
        while maintaining computational efficiency.
        """
        user_interaction_counts = self.sequences_df['userId'].value_counts()
        active_users = user_interaction_counts[user_interaction_counts >= self.min_user_interactions].index
        
        original_count = len(self.sequences_df)
        self.sequences_df = self.sequences_df[self.sequences_df['userId'].isin(active_users)]
        filtered_count = len(self.sequences_df)
        
        logging.info(f"Filtered to {len(active_users):,} users with {self.min_user_interactions}+ interactions")
        logging.info(f"Retained {filtered_count:,} of {original_count:,} total interactions ({filtered_count/original_count*100:.1f}%)")
    
    def _create_id_mappings(self):
        """
        Create mappings between original IDs and tensor indices for efficient processing.
        
        Neural networks work with integer indices rather than arbitrary ID values.
        These mappings allow us to convert between your original user and movie
        IDs and the sequential indices required for embedding layer lookups.
        """
        # Create user ID mappings
        unique_users = sorted(self.sequences_df['userId'].unique())
        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        
        # Create movie ID mappings, ensuring movies exist in both datasets
        # This handles cases where some movies have ratings but no content features
        rated_movies = set(self.sequences_df['movieId'].unique())
        featured_movies = set(self.movies_df['movieId'].unique())
        valid_movies = sorted(rated_movies.intersection(featured_movies))
        
        self.movie_to_idx = {mid: i for i, mid in enumerate(valid_movies)}
        self.idx_to_movie = {i: mid for mid, i in self.movie_to_idx.items()}
        
        # Filter sequences to only include movies with features
        original_seq_count = len(self.sequences_df)
        self.sequences_df = self.sequences_df[self.sequences_df['movieId'].isin(valid_movies)]
        filtered_seq_count = len(self.sequences_df)
        
        logging.info(f"Created mappings: {len(self.user_to_idx):,} users, {len(self.movie_to_idx):,} movies")
        logging.info(f"Filtered sequences: {filtered_seq_count:,} of {original_seq_count:,} have content features")
        
        # Store dimensions for model initialization
        self.num_users = len(self.user_to_idx)
        self.num_movies = len(self.movie_to_idx)
    
    def _prepare_movie_features(self):
        """
        Convert your Stage 4 movie content features into tensor format for training.
        
        This method takes the rich feature set you created during TMDB processing
        and organizes it into the numerical representation required for neural
        network training. The process handles missing values and ensures consistent
        feature scaling across different types of content characteristics.
        """
        # Filter movies dataframe to only include movies in our training set
        valid_movies = list(self.movie_to_idx.keys())
        self.movies_df = self.movies_df[self.movies_df['movieId'].isin(valid_movies)]
        
        # Sort by movie ID to ensure consistent ordering with our index mapping
        self.movies_df = self.movies_df.sort_values('movieId').reset_index(drop=True)
        
        feature_components = []
        
        # Extract numerical features from TMDB metadata
        numerical_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity', 'roi']
        available_numerical = [col for col in numerical_cols if col in self.movies_df.columns]
        if available_numerical:
            numerical_features = self.movies_df[available_numerical].fillna(0).values
            feature_components.append(numerical_features)
            logging.info(f"Added {len(available_numerical)} numerical features")
        
        # Extract one-hot encoded language features
        lang_cols = [col for col in self.movies_df.columns if col.startswith('lang_')]
        if lang_cols:
            language_features = self.movies_df[lang_cols].fillna(0).values
            feature_components.append(language_features)
            logging.info(f"Added {len(lang_cols)} language features")
        
        # Extract categorical features
        categorical_cols = ['is_adult', 'is_independent', 'has_budget', 'has_revenue', 'has_runtime']
        available_categorical = [col for col in categorical_cols if col in self.movies_df.columns]
        if available_categorical:
            categorical_features = self.movies_df[available_categorical].fillna(0).values
            feature_components.append(categorical_features)
            logging.info(f"Added {len(available_categorical)} categorical features")
        
        # Extract text embeddings from your EmbeddingGemma processing
        if 'text_embedding' in self.movies_df.columns:
            # Handle cases where some movies might not have embeddings
            text_embeddings = []
            for _idx, row in self.movies_df.iterrows():
                if row['text_embedding'] is not None and isinstance(row['text_embedding'], list):
                    text_embeddings.append(row['text_embedding'])
                else:
                    # Use zero vector for movies without text embeddings
                    text_embeddings.append([0.0] * 768)  # EmbeddingGemma dimension
            
            text_embeddings = np.array(text_embeddings)
            feature_components.append(text_embeddings)
            logging.info(f"Added text embeddings with dimension {text_embeddings.shape[1]}")
        
        # Combine all feature components into a single matrix
        if feature_components:
            self.movie_features = np.concatenate(feature_components, axis=1)
        else:
            raise ValueError("No movie features found in the dataset")
        
        # Convert to tensor for training
        self.movie_features = torch.FloatTensor(self.movie_features)
        
        logging.info(f"Prepared movie features matrix: {self.movie_features.shape}")
        
        # Create a mapping from movie index to feature row index
        # Since movies_df is already filtered to valid movies, we can map directly
        self.movie_feature_map = {
            self.movie_to_idx[movie_id]: idx for idx, movie_id in enumerate(self.movies_df['movieId'])
        }

    def _create_training_examples(self):
        """
        Generate positive and negative training examples using thumbs ratings.

        This method creates the training pairs that teach your model to distinguish
        between content users enjoy versus content they would avoid. The thumbs
        rating system provides clear positive and negative signals that eliminate
        the ambiguity common in traditional rating systems.
        """
        # Use the extracted method to create examples from all sequences
        positive_examples, negative_examples = self._create_examples_from_sequences(
            self.sequences_df, generate_implicit=True
        )

        self.positive_examples = positive_examples
        self.negative_examples = negative_examples

        logging.info(f"Final training set: {len(positive_examples):,} positive, {len(negative_examples):,} negative")
    
    def _create_examples_from_sequences(self, sequences_df: pd.DataFrame, generate_implicit: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate training examples from a given sequences DataFrame.

        This extracted method allows creating examples from any subset of sequences,
        enabling proper temporal splitting while maintaining the same example
        generation logic.

        Args:
            sequences_df: DataFrame with user interactions and thumbs ratings
            generate_implicit: Whether to generate implicit negative examples

        Returns:
            Tuple of (positive_examples, negative_examples)
        """
        positive_examples = []
        negative_examples = []

        # Process positive examples: thumbs up (1.0) and two thumbs up (2.0)
        positive_ratings = sequences_df[sequences_df['thumbs_rating'] >= 1.0]

        for _, row in positive_ratings.iterrows():
            # Skip if user or movie not in our mappings (shouldn't happen after filtering)
            if row['userId'] not in self.user_to_idx or row['movieId'] not in self.movie_to_idx:
                continue

            user_idx = self.user_to_idx[row['userId']]
            movie_idx = self.movie_to_idx[row['movieId']]
            rating_strength = row['thumbs_rating']  # 1.0 or 2.0

            positive_examples.append({
                'user_idx': user_idx,
                'movie_idx': movie_idx,
                'rating': rating_strength
            })

        # Process explicit negative examples: thumbs down (-1.0)
        negative_ratings = sequences_df[sequences_df['thumbs_rating'] == -1.0]

        for _, row in negative_ratings.iterrows():
            # Skip if user or movie not in our mappings
            if row['userId'] not in self.user_to_idx or row['movieId'] not in self.movie_to_idx:
                continue

            user_idx = self.user_to_idx[row['userId']]
            movie_idx = self.movie_to_idx[row['movieId']]

            negative_examples.append({
                'user_idx': user_idx,
                'movie_idx': movie_idx,
                'rating': -1.0
            })

        logging.info(f"Created {len(positive_examples):,} positive examples from {len(sequences_df):,} sequences")
        logging.info(f"Created {len(negative_examples):,} explicit negative examples")

        # Generate additional implicit negative examples if requested
        if generate_implicit and len(negative_examples) < len(positive_examples) * 0.3:
            self._generate_implicit_negatives_for_examples(positive_examples, negative_examples)

        return positive_examples, negative_examples

    def _generate_implicit_negatives_for_examples(self, positive_examples: List[Dict], negative_examples: List[Dict]):
        """
        Generate implicit negative examples for a given set of positive examples.

        This method is extracted to work with any set of examples, not just the full dataset.
        """
        # Group positive examples by user to understand their preferences
        user_positive_movies = {}
        for example in positive_examples:
            user_idx = example['user_idx']
            if user_idx not in user_positive_movies:
                user_positive_movies[user_idx] = set()
            user_positive_movies[user_idx].add(example['movie_idx'])

        # Generate implicit negatives for each user
        target_negatives = len(positive_examples) * self.negative_sampling_ratio
        current_negatives = len(negative_examples)
        needed_negatives = max(0, target_negatives - current_negatives)

        implicit_count = 0
        all_movie_indices = set(range(self.num_movies))

        for user_idx, positive_movies in user_positive_movies.items():
            if implicit_count >= needed_negatives:
                break

            # Sample movies this user hasn't interacted with
            unrated_movies = all_movie_indices - positive_movies

            if len(unrated_movies) == 0:
                continue

            # Sample a reasonable number of negatives per user
            sample_size = min(len(unrated_movies), max(1, needed_negatives // len(user_positive_movies)))
            sampled_negatives = np.random.choice(list(unrated_movies),
                                                size=sample_size,
                                                replace=False)

            for movie_idx in sampled_negatives:
                negative_examples.append({
                    'user_idx': user_idx,
                    'movie_idx': movie_idx,
                    'rating': 0.0  # Implicit negative marker
                })
                implicit_count += 1

        logging.info(f"Generated {implicit_count:,} additional implicit negative examples")


class TwoTowerDataset(Dataset):
    """
    PyTorch Dataset class that provides training examples for the two-tower model.
    
    This class handles the conversion from your processed training examples into
    the format expected by PyTorch's DataLoader system. It manages the sampling
    of positive and negative examples during training, ensuring that each batch
    contains a balanced mix of preference signals.
    """
    
    def __init__(self, positive_examples: List[Dict], negative_examples: List[Dict],
                 movie_features: torch.Tensor, movie_feature_map: Dict[int, int]):
        """
        Initialize the dataset with your training examples and movie features.
        
        The dataset combines positive examples (thumbs up, two thumbs up) with
        negative examples (thumbs down, implicit negatives) to create training
        pairs that teach the model to distinguish user preferences.
        
        Args:
            positive_examples: List of positive user-movie interactions
            negative_examples: List of negative user-movie interactions  
            movie_features: Tensor containing all movie content features [num_movies, feature_dim]
            movie_feature_map: Mapping from movie ID to feature tensor index
        """
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.movie_features = movie_features
        self.movie_feature_map = movie_feature_map
        
        # Create balanced training pairs by matching each positive with negatives
        self.training_pairs = self._create_training_pairs()
    
    def _create_training_pairs(self) -> List[Dict]:
        """
        Create balanced positive-negative training pairs for contrastive learning.
        
        This method implements the sampling strategy that ensures the model sees
        roughly equal numbers of positive and negative examples during training.
        The balance is crucial for stable learning and prevents the model from
        developing bias toward either positive or negative predictions.
        """
        training_pairs = []

        # Check if we have both positive and negative examples
        if len(self.positive_examples) == 0 or len(self.negative_examples) == 0:
            logging.warning(f"Cannot create training pairs: {len(self.positive_examples)} positive, {len(self.negative_examples)} negative examples")
            return training_pairs

        # For each positive example, create a training pair with a negative example
        for _i, pos_example in enumerate(self.positive_examples):
            # Sample a negative example for contrast
            neg_idx = np.random.randint(0, len(self.negative_examples))
            neg_example = self.negative_examples[neg_idx]

            training_pairs.append({
                'user_idx': pos_example['user_idx'],
                'pos_movie_idx': pos_example['movie_idx'],
                'neg_movie_idx': neg_example['movie_idx'],
                'pos_rating': pos_example['rating'],
                'neg_rating': neg_example['rating']
            })

        return training_pairs
    
    def __len__(self) -> int:
        """Return the total number of training pairs in the dataset."""
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single training example for the model.
        
        This method converts the training pair into the tensor format expected
        by your two-tower model's forward pass. Each example contains a user
        and two movies (one positive, one negative) along with their content features.
        
        Args:
            idx: Index of the training pair to retrieve
            
        Returns:
            Dictionary containing tensors for model training
        """
        pair = self.training_pairs[idx]
        
        # Get movie features for both positive and negative items
        pos_movie_features = self.movie_features[self.movie_feature_map[pair['pos_movie_idx']]]
        neg_movie_features = self.movie_features[self.movie_feature_map[pair['neg_movie_idx']]]
        
        return {
            'user_id': torch.tensor(pair['user_idx'], dtype=torch.long),
            'pos_movie_id': torch.tensor(pair['pos_movie_idx'], dtype=torch.long),
            'neg_movie_id': torch.tensor(pair['neg_movie_idx'], dtype=torch.long),
            'pos_movie_features': pos_movie_features,
            'neg_movie_features': neg_movie_features,
            'pos_rating': torch.tensor(pair['pos_rating'], dtype=torch.float),
            'neg_rating': torch.tensor(pair['neg_rating'], dtype=torch.float)
        }
    
class TwoTowerTrainer:
    """
    Training orchestrator for the two-tower recommendation model.
    
    This class manages the complete training process including data loading,
    loss calculation, optimization, and evaluation. It handles the complex
    coordination between different training components while providing
    monitoring and logging capabilities for training progress.
    """
    
    def __init__(self, model: TwoTowerModel, data_loader: TwoTowerDataLoader,
                 learning_rate: float = 0.001, margin: float = 1.0):
        """
        Initialize the trainer with model and data components.
        
        The trainer combines your two-tower model with the data loading system
        to create a complete training pipeline. It manages the optimization
        process that gradually improves the model's ability to predict user
        preferences through iterative parameter updates.
        
        Args:
            model: Your initialized two-tower model
            data_loader: Prepared data loader with training examples
            learning_rate: Step size for gradient descent optimization
            margin: Margin parameter for ranking loss function
        """
        self.model = model
        self.data_loader = data_loader
        self.margin = margin
        
        # Initialize the optimizer that updates model parameters
        # Adam optimizer provides adaptive learning rates that often work well
        # for recommendation system training
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Track training progress and metrics
        self.training_history = {'epoch': [], 'loss': [], 'metrics': []}
    
    def compute_ranking_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the ranking loss that drives model learning.
        
        This method implements the mathematical loss function we discussed:
        L = max(0, margin - (score_positive - score_negative))
        
        The loss encourages the model to assign higher scores to preferred items
        than to non-preferred items, with a minimum separation defined by the margin.
        
        Args:
            batch: Training batch containing user-item pairs and features
            
        Returns:
            Loss tensor for backpropagation
        """
        # Compute compatibility scores for positive user-item pairs
        pos_scores = self.model(
            batch['user_id'], 
            batch['pos_movie_id'], 
            batch['pos_movie_features']
        )
        
        # Compute compatibility scores for negative user-item pairs
        neg_scores = self.model(
            batch['user_id'], 
            batch['neg_movie_id'], 
            batch['neg_movie_features']
        )
        
        # Apply ranking loss: encourage positive scores > negative scores + margin
        # The torch.relu function implements max(0, x) efficiently
        loss = torch.relu(self.margin - (pos_scores - neg_scores))
        
        return loss.mean()  # Average across the batch
    
    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Execute one complete training epoch over all training data.
        
        An epoch represents one complete pass through your training dataset.
        During each epoch, the model sees every training example once and
        updates its parameters based on the accumulated gradient information.
        
        Args:
            data_loader: PyTorch DataLoader providing training batches
            
        Returns:
            Average loss across all batches in the epoch
        """
        self.model.train()  # Set model to training mode
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            # Clear gradients from previous iteration
            self.optimizer.zero_grad()
            
            # Compute loss for current batch
            loss = self.compute_ranking_loss(batch)
            
            # Backpropagate gradients through the model
            loss.backward()
            
            # Update model parameters using computed gradients
            self.optimizer.step()
            
            # Accumulate loss for monitoring
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches  # Return average loss

    def _evaluate_epoch(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on validation data.

        Args:
            data_loader: PyTorch DataLoader providing validation batches

        Returns:
            Average loss across all validation batches
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch in data_loader:
                loss = self.compute_ranking_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        self.model.train()  # Return to training mode

        # Handle case where validation set is empty
        if num_batches == 0:
            logging.warning("No validation batches found - validation set may be empty")
            return 0.0

        return total_loss / num_batches

    def train(self, num_epochs: int, batch_size: int = 256, 
              validation_split: float = 0.2) -> Dict[str, List]:
        """
        Execute the complete training process over multiple epochs.
        
        This method orchestrates the entire training pipeline including data
        splitting, batch processing, and progress monitoring. It implements
        the iterative optimization process that gradually improves model
        performance through repeated exposure to training examples.
        
        Args:
            num_epochs: Number of complete passes through the training data
            batch_size: Number of training examples processed together
            validation_split: Fraction of data reserved for validation
            
        Returns:
            Training history with loss and metrics over time
        """
        logging.info(f"Starting training for {num_epochs} epochs with batch size {batch_size}")
        
        # Create train/validation split for monitoring overfitting
        train_examples, val_examples = self._create_train_val_split(validation_split)
        
        # Create PyTorch DataLoaders for efficient batch processing
        # Ensure movie_features is a torch.Tensor
        movie_features_tensor = torch.tensor(self.data_loader.movie_features, dtype=torch.float32) \
            if not isinstance(self.data_loader.movie_features, torch.Tensor) \
            else self.data_loader.movie_features

        train_dataset = TwoTowerDataset(
            train_examples['positive'], train_examples['negative'],
            movie_features_tensor, self.data_loader.movie_feature_map
        )

        val_dataset = TwoTowerDataset(
            val_examples['positive'], val_examples['negative'],
            movie_features_tensor, self.data_loader.movie_feature_map
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop: iterate through epochs
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            val_loss = self._evaluate_epoch(val_loader)
            
            # Log progress
            logging.info(f"Epoch {epoch+1}/{num_epochs}: "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Store training history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append({'train': train_loss, 'val': val_loss})
        
        logging.info("Training completed successfully")
        return self.training_history
    
    def create_temporal_splits(self, sequences_df: pd.DataFrame,
                              train_ratio: float = 0.6,
                              val_ratio: float = 0.2,
                              test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits that respect chronological ordering.

        This method implements proper temporal splitting where we use early interactions
        for training, middle interactions for validation, and recent interactions for
        testing. This approach simulates the real-world scenario where we predict
        future user behavior based on historical data.

        The key principle: training data must be temporally earlier than validation
        data, which must be temporally earlier than test data. This ensures that
        our evaluation measures the model's ability to predict future preferences
        based on past behavior, which is exactly what happens in production.

        Args:
            sequences_df: DataFrame with user interactions including timestamps
            train_ratio: Fraction of data for training (earliest interactions)
            val_ratio: Fraction of data for validation (middle interactions)
            test_ratio: Fraction of data for testing (latest interactions)

        Returns:
            Tuple of (train_df, val_df, test_df) with proper temporal ordering
        """

        # Sort all interactions by timestamp to establish temporal ordering
        # This step is crucial - without proper temporal ordering, we can't create
        # meaningful splits that respect causality
        if 'timestamp' in sequences_df.columns:
            sorted_df = sequences_df.sort_values('timestamp').reset_index(drop=True)
        elif 'datetime' in sequences_df.columns:
            sorted_df = sequences_df.sort_values('datetime').reset_index(drop=True)
        else:
            logging.warning("No timestamp column found. Using row order as temporal proxy.")
            sorted_df = sequences_df.reset_index(drop=True)

        # Calculate split indices based on temporal ordering
        total_interactions = len(sorted_df)
        train_end = int(total_interactions * train_ratio)
        val_end = train_end + int(total_interactions * val_ratio)

        # Create temporally-ordered splits
        train_df = sorted_df.iloc[:train_end].copy()
        val_df = sorted_df.iloc[train_end:val_end].copy()
        test_df = sorted_df.iloc[val_end:].copy()

        # Log split statistics to verify proper temporal ordering
        logging.info(f"Temporal split created:")
        logging.info(f"  Training: {len(train_df):,} interactions (earliest)")
        logging.info(f"  Validation: {len(val_df):,} interactions (middle)")
        logging.info(f"  Test: {len(test_df):,} interactions (latest)")

        if 'timestamp' in sorted_df.columns or 'datetime' in sorted_df.columns:
            timestamp_col = 'timestamp' if 'timestamp' in sorted_df.columns else 'datetime'
            logging.info(f"  Training period: {train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}")
            logging.info(f"  Validation period: {val_df[timestamp_col].min()} to {val_df[timestamp_col].max()}")
            logging.info(f"  Test period: {test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}")

        return train_df, val_df, test_df

    def analyze_split_overlap(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                             test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze user and item overlap between temporal splits to understand evaluation setup.

        This analysis helps verify that our temporal splitting approach creates meaningful
        evaluation conditions. Ideal temporal splits should have some user overlap (so we
        can evaluate on known users) but limited item overlap in the recent interactions
        (so we're testing the model's ability to recommend newer content).

        Args:
            train_df: Training set interactions
            val_df: Validation set interactions
            test_df: Test set interactions

        Returns:
            Dictionary containing overlap statistics and insights
        """
        analysis = {}

        # Analyze user overlap across splits
        train_users = set(train_df['userId'].unique())
        val_users = set(val_df['userId'].unique())
        test_users = set(test_df['userId'].unique())

        analysis['user_stats'] = {
            'train_users': len(train_users),
            'val_users': len(val_users),
            'test_users': len(test_users),
            'train_val_overlap': len(train_users & val_users),
            'train_test_overlap': len(train_users & test_users),
            'val_test_overlap': len(val_users & test_users),
            'all_three_overlap': len(train_users & val_users & test_users)
        }

        # Analyze item overlap across splits
        train_items = set(train_df['movieId'].unique())
        val_items = set(val_df['movieId'].unique())
        test_items = set(test_df['movieId'].unique())

        analysis['item_stats'] = {
            'train_items': len(train_items),
            'val_items': len(val_items),
            'test_items': len(test_items),
            'train_val_overlap': len(train_items & val_items),
            'train_test_overlap': len(train_items & test_items),
            'val_test_overlap': len(val_items & test_items),
            'all_three_overlap': len(train_items & val_items & test_items)
        }

        # Calculate overlap percentages for interpretation
        analysis['user_overlap_pct'] = {
            'val_users_in_train': len(train_users & val_users) / len(val_users) * 100,
            'test_users_in_train': len(train_users & test_users) / len(test_users) * 100,
            'test_users_in_val': len(val_users & test_users) / len(test_users) * 100
        }

        analysis['item_overlap_pct'] = {
            'val_items_in_train': len(train_items & val_items) / len(val_items) * 100,
            'test_items_in_train': len(train_items & test_items) / len(test_items) * 100,
            'test_items_in_val': len(val_items & test_items) / len(test_items) * 100
        }

        # Temporal statistics if timestamp columns exist
        if 'timestamp' in train_df.columns:
            analysis['temporal_stats'] = {
                'train_period': (train_df['timestamp'].min(), train_df['timestamp'].max()),
                'val_period': (val_df['timestamp'].min(), val_df['timestamp'].max()),
                'test_period': (test_df['timestamp'].min(), test_df['timestamp'].max()),
                'train_duration_days': (train_df['timestamp'].max() - train_df['timestamp'].min()) / (24*3600),
                'val_duration_days': (val_df['timestamp'].max() - val_df['timestamp'].min()) / (24*3600),
                'test_duration_days': (test_df['timestamp'].max() - test_df['timestamp'].min()) / (24*3600)
            }

        logging.info("Split overlap analysis:")
        logging.info(f"  User overlap - Val/Train: {analysis['user_overlap_pct']['val_users_in_train']:.1f}%, "
                    f"Test/Train: {analysis['user_overlap_pct']['test_users_in_train']:.1f}%")
        logging.info(f"  Item overlap - Val/Train: {analysis['item_overlap_pct']['val_items_in_train']:.1f}%, "
                    f"Test/Train: {analysis['item_overlap_pct']['test_items_in_train']:.1f}%")

        return analysis

    def load_and_split_data(self, sequences_path: str, movies_path: str,
                           train_ratio: float = 0.6, val_ratio: float = 0.2,
                           test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Complete data loading and temporal splitting pipeline with overlap analysis.

        This method combines data loading, temporal splitting, and overlap analysis
        into a single pipeline that provides both the split datasets and insights
        about their characteristics. This is the recommended way to prepare data
        for model training and evaluation.

        Args:
            sequences_path: Path to the user interaction sequences
            movies_path: Path to the movie content features
            train_ratio: Fraction of earliest interactions for training
            val_ratio: Fraction of middle interactions for validation
            test_ratio: Fraction of latest interactions for testing

        Returns:
            Tuple containing (train_df, val_df, test_df, overlap_analysis)
        """
        logging.info("Loading data for temporal splitting...")

        # Load the interaction sequences
        sequences_df = pd.read_parquet(sequences_path)
        logging.info(f"Loaded {len(sequences_df):,} interactions from {sequences_path}")

        # Validate ratios sum to 1.0
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        # Create temporal splits
        train_df, val_df, test_df = self.create_temporal_splits(
            sequences_df, train_ratio, val_ratio, test_ratio
        )

        # Analyze the splits to understand evaluation setup
        overlap_analysis = self.analyze_split_overlap(train_df, val_df, test_df)

        # Load movie features for completeness (though not directly returned)
        if movies_path and Path(movies_path).exists():
            movies_df = pd.read_parquet(movies_path)
            logging.info(f"Loaded {len(movies_df):,} movies from {movies_path}")
        else:
            logging.warning(f"Movie features file not found: {movies_path}")

        return train_df, val_df, test_df, overlap_analysis

    def _create_train_val_split(self, validation_split: float) -> Tuple[Dict, Dict]:
        """
        Create temporal train-validation split that respects chronological ordering.

        This method implements proper temporal splitting by first splitting the sequences
        by datetime, then creating training examples from each temporal split separately.
        This ensures both splits have proper positive/negative ratios from their respective
        time periods.

        Args:
            validation_split: Fraction of recent interactions reserved for validation

        Returns:
            Tuple containing train and validation example dictionaries
        """
        logging.info("Creating temporal train-validation split...")

        # Split sequences temporally FIRST (sequences have datetime information)
        train_ratio = 1.0 - validation_split
        val_ratio = validation_split

        train_sequences, val_sequences, _ = self.create_temporal_splits(
            self.data_loader.sequences_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=0.0
        )

        logging.info(f"Split sequences temporally: {len(train_sequences):,} train, {len(val_sequences):,} validation")

        # Create examples from each temporal split separately
        # This preserves the temporal ordering while ensuring balanced positive/negative ratios
        train_positive, train_negative = self.data_loader._create_examples_from_sequences(
            train_sequences, generate_implicit=True
        )

        val_positive, val_negative = self.data_loader._create_examples_from_sequences(
            val_sequences, generate_implicit=False  # Don't generate implicit negatives for validation
        )

        logging.info(f"Train split: {len(train_positive):,} positive, {len(train_negative):,} negative")
        logging.info(f"Val split: {len(val_positive):,} positive, {len(val_negative):,} negative")

        # Validate that both splits have reasonable distributions
        if len(val_positive) == 0:
            logging.warning("Validation split has no positive examples! This may indicate temporal distribution issues.")
        if len(train_positive) == 0:
            logging.error("Training split has no positive examples! This is a critical error.")

        train_pos_ratio = len(train_positive) / (len(train_positive) + len(train_negative)) if train_positive or train_negative else 0
        val_pos_ratio = len(val_positive) / (len(val_positive) + len(val_negative)) if val_positive or val_negative else 0

        logging.info(f"Positive ratios - Train: {train_pos_ratio:.3f}, Val: {val_pos_ratio:.3f}")

        return (
            {'positive': train_positive, 'negative': train_negative},
            {'positive': val_positive, 'negative': val_negative}
        )
    
class TwoTowerEvaluator:
    """
    Comprehensive evaluation system for two-tower recommendation models.
    
    This class implements the specialized metrics needed to assess recommendation
    system performance, focusing on retrieval quality and ranking effectiveness.
    The evaluation approach accounts for the unique characteristics of recommendation
    tasks where users interact with ranked lists rather than individual predictions.
    """
    
    def __init__(self, model: TwoTowerModel, data_loader: TwoTowerDataLoader):
        """
        Initialize the evaluator with your trained model and data.
        
        The evaluator combines your trained two-tower model with the processed
        data to compute recommendation quality metrics. It handles the complex
        process of generating recommendations for test users and comparing them
        against actual user preferences.
        
        Args:
            model: Your trained two-tower recommendation model
            data_loader: Data loader containing test examples and movie features
        """
        self.model = model
        self.data_loader = data_loader
        self.model.eval()  # Set model to evaluation mode
    
    def compute_recall_at_k(self, user_recommendations: Dict[int, List[int]], 
                           user_ground_truth: Dict[int, List[int]], 
                           k_values: List[int] = [10, 50, 100]) -> Dict[int, float]:
        """
        Calculate recall at different cutoff points for recommendation lists.
        
        Recall@K measures what fraction of movies that users actually liked
        appear within the top-K recommendations. This metric directly reflects
        the model's ability to identify relevant content for users, which is
        the primary goal of the retrieval stage in your recommendation pipeline.
        
        The mathematical definition: Recall@K = |relevant ∩ top-K| / |relevant|
        where 'relevant' represents movies the user actually liked in the test set.
        
        Args:
            user_recommendations: Dict mapping user_idx to ranked recommendation lists
            user_ground_truth: Dict mapping user_idx to lists of actually liked movies
            k_values: List of cutoff points to evaluate (e.g., top-10, top-50, top-100)
            
        Returns:
            Dictionary mapping each K value to average recall across all users
        """
        recall_results = {k: [] for k in k_values}
        
        for user_idx, ground_truth_items in user_ground_truth.items():
            if user_idx not in user_recommendations or len(ground_truth_items) == 0:
                continue  # Skip users without recommendations or ground truth
            
            recommended_items = user_recommendations[user_idx]
            
            # Calculate recall at each K value for this user
            for k in k_values:
                top_k_recommendations = set(recommended_items[:k])
                relevant_items = set(ground_truth_items)
                
                # Count how many relevant items appear in top-K recommendations
                hits = len(top_k_recommendations.intersection(relevant_items))
                
                # Calculate recall: hits / total_relevant_items
                user_recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
                recall_results[k].append(user_recall)
        
        # Average recall across all users for each K value
        avg_recall = {k: float(np.mean(scores)) for k, scores in recall_results.items()}

        return avg_recall
    
    def compute_coverage(self, user_recommendations: Dict[int, List[int]], 
                        k: int = 100) -> float:
        """
        Measure catalog coverage of the recommendation system.
        
        Coverage evaluates how much of your movie catalog the model actually
        recommends to users. Low coverage suggests the model focuses too heavily
        on popular items, potentially creating filter bubbles where users only
        see mainstream content. High coverage indicates the model can surface
        diverse content across your entire catalog.
        
        This metric becomes especially important for movie recommendation systems
        because users often want to discover new and interesting films rather
        than just receiving obvious popular choices.
        
        Args:
            user_recommendations: Dict mapping user_idx to recommendation lists
            k: Cutoff point for measuring coverage (typically 100)
            
        Returns:
            Fraction of total catalog covered by recommendations
        """
        # Collect all unique items recommended across all users
        all_recommended_items = set()
        for _user_idx, recommendations in user_recommendations.items():
            all_recommended_items.update(recommendations[:k])
        
        # Calculate coverage as fraction of total catalog
        total_items = self.data_loader.num_movies
        coverage = len(all_recommended_items) / total_items
        
        return coverage
    
    def generate_recommendations_for_user(self, user_idx: int, k: int = 100,
                                        exclude_seen: bool = True) -> List[int]:
        """
        Generate ranked movie recommendations for a specific user.
        
        This method demonstrates how your trained two-tower model generates
        actual recommendations in practice. It computes the user's embedding
        once, then calculates similarity scores against all movie embeddings
        to produce a ranked list of candidates.
        
        Args:
            user_idx: Index of the user to generate recommendations for
            k: Number of recommendations to return
            exclude_seen: Whether to filter out movies the user has already rated
            
        Returns:
            List of movie indices ranked by predicted preference
        """
        # Get user embedding from the trained model
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        user_embedding = self.model.get_user_embeddings(user_tensor)
        
        # Get all movie embeddings efficiently
        all_movie_indices = list(range(self.data_loader.num_movies))
        movie_tensors = torch.tensor(all_movie_indices, dtype=torch.long)

        # Ensure movie_features is a torch.Tensor
        movie_features_tensor = torch.tensor(self.data_loader.movie_features, dtype=torch.float32) \
            if not isinstance(self.data_loader.movie_features, torch.Tensor) \
            else self.data_loader.movie_features

        movie_embeddings = self.model.get_movie_embeddings(
            movie_tensors, movie_features_tensor
        )
        
        # Calculate similarity scores between user and all movies
        similarity_scores = torch.matmul(user_embedding, movie_embeddings.T).squeeze()
        
        # Convert to numpy for easier manipulation
        scores_np = similarity_scores.detach().cpu().numpy()
        
        # Create list of (movie_idx, score) pairs
        movie_scores = list(zip(all_movie_indices, scores_np))

        # Sort by score in descending order (highest similarity first)
        movie_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract just the movie indices, ranked by predicted preference
        ranked_movies = [movie_idx for movie_idx, _score in movie_scores]
        
        # Filter out movies the user has already seen if requested
        if exclude_seen:
            seen_movies = self._get_user_seen_movies(user_idx)
            ranked_movies = [mid for mid in ranked_movies if mid not in seen_movies]
        
        return ranked_movies[:k]
    
    
    def evaluate_model_performance(self, test_examples: Dict[str, List[Dict]], 
                             k_values: List[int] = [10, 50, 100]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for the trained model.
        
        This method orchestrates the complete evaluation process, generating
        recommendations for test users and computing the metrics that measure
        recommendation quality. The evaluation respects the temporal split
        we used during training, ensuring that we test the model's ability
        to predict future user preferences based on past behavior.
        
        Args:
            test_examples: Dictionary containing positive and negative test examples
            k_values: List of cutoff points for evaluation metrics
            
        Returns:
            Dictionary containing all computed evaluation metrics
        """
        logging.info("Starting comprehensive model evaluation...")
        
        # Extract test users and their ground truth preferences
        test_users = self._extract_test_users(test_examples)
        user_ground_truth = self._create_ground_truth_mapping(test_examples['positive'])
        
        logging.info(f"Evaluating on {len(test_users)} test users")
        
        # Generate recommendations for all test users
        user_recommendations = {}
        for user_idx in test_users:
            try:
                recommendations = self.generate_recommendations_for_user(
                    user_idx, k=max(k_values), exclude_seen=True
                )
                user_recommendations[user_idx] = recommendations
            except Exception as e:
                logging.warning(f"Failed to generate recommendations for user {user_idx}: {e}")
                continue
        
        # Compute recall metrics at different cutoff points
        recall_results = self.compute_recall_at_k(
            user_recommendations, user_ground_truth, k_values
        )
        
        # Compute coverage metrics
        coverage_results = {}
        for k in k_values:
            coverage = self.compute_coverage(user_recommendations, k)
            coverage_results[f'coverage@{k}'] = coverage
        
        # Combine all metrics into a comprehensive results dictionary
        evaluation_results = {}

        # Add recall metrics (use _at_ instead of @ for MLflow compatibility)
        for k, recall_value in recall_results.items():
            evaluation_results[f'recall_at_{k}'] = recall_value

        # Add coverage metrics (rename to use _at_ for MLflow)
        for key, value in coverage_results.items():
            # Replace @ with _at_ in coverage metric names
            new_key = key.replace('@', '_at_')
            evaluation_results[new_key] = value
        
        # Add summary statistics
        evaluation_results['num_test_users'] = len(test_users)
        evaluation_results['avg_recommendations_per_user'] = np.mean([
            len(recs) for recs in user_recommendations.values()
        ])
        
        # Log key results
        logging.info("Evaluation Results:")
        for metric, value in evaluation_results.items():
            if isinstance(value, float):
                logging.info(f"  {metric}: {value:.4f}")
            else:
                logging.info(f"  {metric}: {value}")
        
        return evaluation_results

    def _extract_test_users(self, test_examples: Dict[str, List[Dict]]) -> List[int]:
        """Extract unique user indices from test examples."""
        test_users = set()
        for example in test_examples['positive'] + test_examples['negative']:
            test_users.add(example['user_idx'])
        return list(test_users)

    def _create_ground_truth_mapping(self, positive_examples: List[Dict]) -> Dict[int, List[int]]:
        """Create mapping from users to movies they actually liked in test set."""
        user_ground_truth = {}
        for example in positive_examples:
            user_idx = example['user_idx']
            movie_idx = example['movie_idx']
            
            if user_idx not in user_ground_truth:
                user_ground_truth[user_idx] = []
            user_ground_truth[user_idx].append(movie_idx)
        
        return user_ground_truth

    def _get_user_seen_movies(self, user_idx: int) -> set:
        """Get set of movies this user has already rated (to exclude from recommendations)."""
        seen_movies = set()
        
        # Look through all training examples to find this user's rated movies
        all_examples = self.data_loader.positive_examples + self.data_loader.negative_examples
        for example in all_examples:
            if example['user_idx'] == user_idx:
                seen_movies.add(example['movie_idx'])
        
        return seen_movies