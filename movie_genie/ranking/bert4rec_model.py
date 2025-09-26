"""
BERT4Rec Implementation for Sequential Movie Recommendation

This module implements BERT4Rec (Bidirectional Encoder Representations from 
Transformers for Recommendation) adapted for movie recommendation with rich
content features. The model learns temporal user preference patterns through
bidirectional attention mechanisms while incorporating TMDB metadata and
text embeddings for enhanced item understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import random
from torch.utils.data import Dataset, DataLoader


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for BERT4Rec.
    
    This implements the core attention computation that enables the model
    to focus on different aspects of user interaction sequences. Multiple
    attention heads allow the model to attend to different types of patterns
    simultaneously - some heads might focus on recent interactions, others
    on thematic consistency, and others on content similarity.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Initialize multi-head attention with specified dimensions.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of parallel attention heads
            dropout_rate: Dropout probability for regularization
        """
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for queries, keys, and values
        # These transform input representations into attention computation space
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)  
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection to combine attention heads
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Scale factor for numerical stability
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head attention over input sequences.
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_dim]
            key: Key tensor [batch_size, seq_len, hidden_dim]  
            value: Value tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Attention output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = query.shape[:2]
        
        # Linear transformations and reshape for multi-head computation
        # Shape: [batch_size, seq_len, num_heads, head_dim]
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2) 
        V = V.transpose(1, 2)
        
        # Compute attention scores
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        self.scale = self.scale.to(query.device)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        # Shape: [batch_size, seq_len, hidden_dim]
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)
        
        return self.W_o(attention_output)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network for transformer blocks.
    
    This component applies non-linear transformations to each position
    in the sequence independently. The two-layer design with ReLU activation
    enables the model to learn complex mappings from attention outputs
    to final representations.
    """
    
    def __init__(self, hidden_dim: int, feed_forward_dim: int, dropout_rate: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            hidden_dim: Input/output dimension
            feed_forward_dim: Hidden dimension of feed-forward layer
            dropout_rate: Dropout probability
        """
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(hidden_dim, feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward transformation."""
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer block combining attention and feed-forward layers.
    
    This implements the core building block of BERT4Rec, combining multi-head
    attention with position-wise feed-forward networks. The residual connections
    and layer normalization ensure stable training and effective information flow
    through the network depth.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, feed_forward_dim: int, 
                 dropout_rate: float = 0.1):
        """
        Initialize transformer block with attention and feed-forward components.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            feed_forward_dim: Dimension of feed-forward hidden layer
            dropout_rate: Dropout probability
        """
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.feed_forward = FeedForward(hidden_dim, feed_forward_dim, dropout_rate)
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Transformed representations [batch_size, seq_len, hidden_dim]
        """
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(x, x, x, mask)
        x = self.ln1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_output))
        
        return x


class BERT4RecModel(nn.Module):
    """
    Complete BERT4Rec model for sequential movie recommendation.
    
    This model learns bidirectional representations of user interaction sequences,
    incorporating rich content features from your TMDB processing and text embeddings.
    The architecture enables sophisticated understanding of user preference evolution
    while leveraging content-based similarity for enhanced recommendation quality.
    """
    
    def __init__(self,
                 num_items: int,
                 content_feature_dim: int,
                 max_seq_len: int = 50,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1):
        """
        Initialize BERT4Rec model with content feature integration.
        
        Args:
            num_items: Number of unique movies in the dataset
            content_feature_dim: Dimension of movie content features
            max_seq_len: Maximum length of interaction sequences
            hidden_dim: Dimension of hidden representations  
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads per block
            dropout_rate: Dropout probability for regularization
        """
        super(BERT4RecModel, self).__init__()
        
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # Item embeddings that learn collaborative filtering patterns
        # These capture latent factors beyond explicit content features
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        # +2 for padding (0) and mask (num_items + 1) tokens
        
        # Position embeddings to encode sequence order information
        # These help the model understand temporal relationships
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Content feature projection to integrate TMDB and text features
        # This allows the model to use both learned and explicit item characteristics
        self.content_projection = nn.Linear(content_feature_dim, hidden_dim)
        
        # Transformer blocks for bidirectional sequence modeling
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for final representations
        self.ln_final = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights for stable training
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using appropriate strategies for transformers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, item_seq: torch.Tensor, content_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through BERT4Rec model.
        
        Args:
            item_seq: Sequence of item IDs [batch_size, seq_len]
            content_features: Content features for items [batch_size, seq_len, content_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Sequence representations [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = item_seq.shape
        
        # Generate position indices for positional encoding
        position_ids = torch.arange(seq_len, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # Combine item, position, and content embeddings
        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)
        content_emb = self.content_projection(content_features)
        
        # Sum embeddings to create unified item representations
        # This integration allows the model to use all available information
        embeddings = item_emb + position_emb + content_emb
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, mask)
        
        # Apply final layer normalization
        sequence_output = self.ln_final(hidden_states)
        
        return sequence_output
    
    def predict_masked_items(self, item_seq: torch.Tensor, content_features: torch.Tensor,
                           candidate_items: torch.Tensor, candidate_features: torch.Tensor,
                           mask_positions: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for candidate items at masked positions.
        
        This method implements the core prediction task for BERT4Rec: given
        a sequence with some items masked, predict which items should fill
        the masked positions. During inference, this becomes ranking candidates
        from your two-tower retrieval system.
        
        Args:
            item_seq: Input sequence with masked positions [batch_size, seq_len]
            content_features: Content features for sequence items [batch_size, seq_len, content_dim]
            candidate_items: Candidate item IDs [batch_size, num_candidates]
            candidate_features: Content features for candidates [batch_size, num_candidates, content_dim]
            mask_positions: Positions of masked items [batch_size, num_masked]
            
        Returns:
            Prediction scores [batch_size, num_masked, num_candidates]
        """
        # Get sequence representations
        sequence_output = self.forward(item_seq, content_features)
        
        # Extract representations at masked positions
        batch_size = sequence_output.shape[0]
        mask_output = sequence_output[torch.arange(batch_size).unsqueeze(1), mask_positions]
        
        # Get candidate item representations
        candidate_emb = self.item_embedding(candidate_items)
        candidate_content_emb = self.content_projection(candidate_features)
        candidate_representations = candidate_emb + candidate_content_emb
        
        # Compute similarity scores between masked positions and candidates
        # Shape: [batch_size, num_masked, num_candidates]
        scores = torch.matmul(mask_output, candidate_representations.transpose(-2, -1))
        
        return scores
    
class BERT4RecDataset(Dataset):
    """
    Dataset class for BERT4Rec training with masking strategy.
    
    This dataset implements the masked item prediction training strategy that
    teaches BERT4Rec to understand user preferences through bidirectional context.
    It integrates with your two-tower candidate generation and rich content features.
    """
    
    def __init__(self, 
                 user_sequences: Dict[int, List[Dict]],
                 movie_features: torch.Tensor,
                 movie_feature_map: Dict[int, int],
                 max_seq_len: int = 50,
                 mask_prob: float = 0.15,
                 num_items: int = None):
        """
        Initialize dataset with user interaction sequences and masking parameters.
        
        Args:
            user_sequences: Dict mapping user_idx to list of interaction records
            movie_features: Tensor of movie content features
            movie_feature_map: Mapping from movie_idx to feature tensor index
            max_seq_len: Maximum sequence length for truncation/padding
            mask_prob: Probability of masking each item during training
            num_items: Total number of items (for mask token ID)
        """
        self.user_sequences = user_sequences
        self.movie_features = movie_features
        self.movie_feature_map = movie_feature_map
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.num_items = num_items
        
        # Special token IDs
        self.pad_token = 0
        self.mask_token = num_items + 1 if num_items else 1
        
        # Convert sequences to training examples
        self.examples = self._create_training_examples()
        
    def _create_training_examples(self) -> List[Dict]:
        """
        Convert user sequences into training examples with proper formatting.
        
        This method processes your temporal user interaction sequences into
        the format needed for BERT4Rec training. It handles sequence truncation,
        padding, and prepares the data for masking during training.
        """
        examples = []
        
        for user_idx, interactions in self.user_sequences.items():
            # Sort interactions by timestamp for temporal ordering
            if interactions and 'timestamp' in interactions[0]:
                interactions = sorted(interactions, key=lambda x: x['timestamp'])
            
            # Extract item sequence and ratings
            item_sequence = [interaction['movie_idx'] for interaction in interactions]
            rating_sequence = [interaction['rating'] for interaction in interactions]
            
            # Skip users with very short sequences
            if len(item_sequence) < 3:
                continue
            
            # Create sliding windows for longer sequences
            if len(item_sequence) <= self.max_seq_len:
                sequences = [item_sequence]
                ratings = [rating_sequence]
            else:
                # Create overlapping windows to use all data
                sequences = []
                ratings = []
                step_size = self.max_seq_len // 2
                for start in range(0, len(item_sequence) - self.max_seq_len + 1, step_size):
                    sequences.append(item_sequence[start:start + self.max_seq_len])
                    ratings.append(rating_sequence[start:start + self.max_seq_len])
            
            # Convert sequences to training examples
            for seq, rat in zip(sequences, ratings):
                examples.append({
                    'user_idx': user_idx,
                    'item_sequence': seq,
                    'rating_sequence': rat
                })
        
        logging.info(f"Created {len(examples)} training sequences from {len(self.user_sequences)} users")
        return examples
    
    def _apply_masking(self, item_sequence: List[int], rating_sequence: List[float]) -> Tuple[List[int], List[int], List[int]]:
        """
        Apply masking strategy for BERT4Rec training.
        
        This method implements the core masking strategy that teaches bidirectional
        understanding. We mask items that received positive ratings (thumbs up or
        two thumbs up) to train the model to predict user preferences based on
        surrounding context.
        
        Args:
            item_sequence: Original sequence of item IDs
            rating_sequence: Corresponding ratings for items
            
        Returns:
            Tuple of (masked_sequence, original_items, mask_positions)
        """
        masked_sequence = item_sequence.copy()
        original_items = []
        mask_positions = []
        
        for i, (item_id, rating) in enumerate(zip(item_sequence, rating_sequence)):
            # Only consider masking positively rated items
            # This focuses learning on predicting user preferences
            if rating >= 1.0 and random.random() < self.mask_prob:
                mask_positions.append(i)
                original_items.append(item_id)
                masked_sequence[i] = self.mask_token
        
        # Ensure at least one item is masked for training
        if not mask_positions:
            # Find a positive rating to mask
            positive_positions = [i for i, r in enumerate(rating_sequence) if r >= 1.0]
            if positive_positions:
                pos = random.choice(positive_positions)
                mask_positions.append(pos)
                original_items.append(item_sequence[pos])
                masked_sequence[pos] = self.mask_token
        
        return masked_sequence, original_items, mask_positions
    
    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad or truncate sequence to max_seq_len."""
        if len(sequence) >= self.max_seq_len:
            return sequence[:self.max_seq_len]
        else:
            return sequence + [self.pad_token] * (self.max_seq_len - len(sequence))
    
    def __len__(self) -> int:
        """Return total number of training examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example with masking applied.
        
        Args:
            idx: Index of training example
            
        Returns:
            Dictionary containing masked sequence, targets, and content features
        """
        example = self.examples[idx]
        item_seq = example['item_sequence']
        rating_seq = example['rating_sequence']
        
        # Apply masking strategy
        masked_seq, target_items, mask_positions = self._apply_masking(item_seq, rating_seq)
        
        # Pad sequences to consistent length
        padded_masked_seq = self._pad_sequence(masked_seq)
        padded_original_seq = self._pad_sequence(item_seq)
        
        # Get content features for the sequence items
        content_features = []
        for item_idx in padded_masked_seq:
            if item_idx == self.pad_token or item_idx == self.mask_token:
                # Use zero features for special tokens
                content_features.append(torch.zeros(self.movie_features.shape[1]))
            else:
                feature_idx = self.movie_feature_map.get(item_idx, 0)
                content_features.append(self.movie_features[feature_idx])
        
        content_features = torch.stack(content_features)
        
        # Prepare targets and positions for loss computation
        # Pad mask positions and targets to consistent length for batching
        max_masks = max(1, int(self.max_seq_len * self.mask_prob))
        
        padded_mask_positions = mask_positions + [-1] * (max_masks - len(mask_positions))
        padded_target_items = target_items + [0] * (max_masks - len(target_items))
        
        return {
            'masked_sequence': torch.tensor(padded_masked_seq, dtype=torch.long),
            'original_sequence': torch.tensor(padded_original_seq, dtype=torch.long),
            'content_features': content_features,
            'mask_positions': torch.tensor(padded_mask_positions[:max_masks], dtype=torch.long),
            'target_items': torch.tensor(padded_target_items[:max_masks], dtype=torch.long),
            'num_masks': torch.tensor(len(mask_positions), dtype=torch.long),
            'user_idx': torch.tensor(example['user_idx'], dtype=torch.long)
        }


class BERT4RecDataLoader:
    """
    Data loading and preprocessing pipeline for BERT4Rec training.
    
    This class handles the complete data preparation pipeline, from reading
    your processed parquet files to creating user sequences suitable for
    BERT4Rec training. It integrates with your existing two-tower data
    and temporal splitting methodology.
    """
    
    def __init__(self,
                 sequences_path: str,
                 movies_path: str, 
                 max_seq_len: int = 50,
                 min_seq_len: int = 3):
        """
        Initialize data loader with paths to processed data files.
        
        Args:
            sequences_path: Path to user interaction sequences
            movies_path: Path to movies with content features
            max_seq_len: Maximum sequence length for training
            min_seq_len: Minimum sequence length for valid training examples
        """
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        # Load processed data from your pipeline
        logging.info("Loading data for BERT4Rec training...")
        self.sequences_df = pd.read_parquet(sequences_path)
        self.movies_df = pd.read_parquet(movies_path)
        
        logging.info(f"Loaded {len(self.sequences_df):,} interactions and {len(self.movies_df):,} movies")
        
        # Create ID mappings consistent with your two-tower system
        self._create_id_mappings()
        
        # Prepare movie content features
        self._prepare_movie_features()
        
        # Create user sequences for BERT4Rec training
        self._create_user_sequences()
    
    def _create_id_mappings(self):
        """Create consistent ID mappings with your two-tower system."""
        # Create user mappings
        unique_users = sorted(self.sequences_df['userId'].unique())
        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        
        # Create movie mappings - only include movies with both ratings and features
        rated_movies = set(self.sequences_df['movieId'].unique())
        featured_movies = set(self.movies_df['movieId'].unique())
        valid_movies = sorted(rated_movies.intersection(featured_movies))
        
        self.movie_to_idx = {mid: i+1 for i, mid in enumerate(valid_movies)}  # +1 to reserve 0 for padding
        self.idx_to_movie = {i: mid for mid, i in self.movie_to_idx.items()}
        
        # Filter sequences to valid movies
        self.sequences_df = self.sequences_df[self.sequences_df['movieId'].isin(valid_movies)]
        
        self.num_users = len(self.user_to_idx)
        self.num_movies = len(self.movie_to_idx)
        
        logging.info(f"Created mappings: {self.num_users} users, {self.num_movies} movies")
    
    def _prepare_movie_features(self):
        """Prepare movie content features matching your two-tower system."""
        # Filter and sort movies to match ID mapping order
        valid_movies = list(self.movie_to_idx.keys())
        self.movies_df = self.movies_df[self.movies_df['movieId'].isin(valid_movies)]
        self.movies_df = self.movies_df.sort_values('movieId').reset_index(drop=True)
        
        # Extract feature components (matching your two-tower implementation)
        feature_components = []
        
        # Numerical features
        numerical_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity', 'roi']
        available_numerical = [col for col in numerical_cols if col in self.movies_df.columns]
        if available_numerical:
            numerical_features = self.movies_df[available_numerical].fillna(0).values
            feature_components.append(numerical_features)
        
        # Language features
        lang_cols = [col for col in self.movies_df.columns if col.startswith('lang_')]
        if lang_cols:
            language_features = self.movies_df[lang_cols].fillna(0).values
            feature_components.append(language_features)
        
        # Categorical features
        categorical_cols = ['is_adult', 'is_independent', 'has_budget', 'has_revenue', 'has_runtime']
        available_categorical = [col for col in categorical_cols if col in self.movies_df.columns]
        if available_categorical:
            categorical_features = self.movies_df[available_categorical].fillna(0).values
            feature_components.append(categorical_features)
        
        # Text embeddings
        if 'text_embedding' in self.movies_df.columns:
            text_embeddings = []
            for _, row in self.movies_df.iterrows():
                if row['text_embedding'] is not None and isinstance(row['text_embedding'], list):
                    text_embeddings.append(row['text_embedding'])
                else:
                    text_embeddings.append([0.0] * 768)  # EmbeddingGemma dimension
            text_embeddings = np.array(text_embeddings)
            feature_components.append(text_embeddings)
        
        # Combine all features
        if feature_components:
            # Add padding row for special tokens at index 0
            combined_features = np.concatenate(feature_components, axis=1)
            padding_row = np.zeros((1, combined_features.shape[1]))
            self.movie_features = torch.FloatTensor(np.vstack([padding_row, combined_features]))
        else:
            raise ValueError("No movie features found")
        
        # Create movie feature mapping
        self.movie_feature_map = {0: 0}  # Padding token maps to index 0
        for i, movie_id in enumerate(sorted(self.movie_to_idx.keys())):
            movie_idx = self.movie_to_idx[movie_id]
            self.movie_feature_map[movie_idx] = i + 1  # +1 because we added padding row
        
        logging.info(f"Prepared movie features: {self.movie_features.shape}")
    
    def _create_user_sequences(self):
        """Create user interaction sequences for BERT4Rec training."""
        self.user_sequences = {}
        
        # Process each user's interactions into sequences
        for user_id in self.sequences_df['userId'].unique():
            user_data = self.sequences_df[self.sequences_df['userId'] == user_id]
            
            # Sort by timestamp if available
            if 'timestamp' in user_data.columns:
                user_data = user_data.sort_values('timestamp')
            
            # Convert to sequence format
            interactions = []
            for _, row in user_data.iterrows():
                interactions.append({
                    'movie_idx': self.movie_to_idx[row['movieId']],
                    'rating': row['rating'],
                    'timestamp': row.get('timestamp', 0)
                })
            
            # Only include users with sufficient interactions
            if len(interactions) >= self.min_seq_len:
                user_idx = self.user_to_idx[user_id]
                self.user_sequences[user_idx] = interactions
        
        logging.info(f"Created sequences for {len(self.user_sequences)} users")