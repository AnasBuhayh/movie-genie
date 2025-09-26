"""
BERT4Rec Training Script for Sequential Movie Recommendation

This script implements the complete BERT4Rec training pipeline, designed to integrate
with your existing DVC workflow and two-tower retrieval system. It handles temporal
sequence creation, bidirectional masking, content feature integration, and evaluation
against both individual sequential modeling performance and combined system metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
import argparse
from typing import Dict, Any, List, Tuple
import pickle

# Import BERT4Rec components we just built
import sys
sys.path.append('.')
from movie_genie.ranking.bert4rec_model import (
    BERT4RecModel, BERT4RecDataset, BERT4RecDataLoader
)

# Import two-tower components for integrated evaluation
from movie_genie.retrieval.two_tower_model import (
    TwoTowerModel, TwoTowerDataLoader as TwoTowerDataLoader
)

def setup_logging() -> None:
    """Configure comprehensive logging for BERT4Rec training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load BERT4Rec configuration parameters from YAML file.
    
    The configuration centralizes all hyperparameters, training settings, and
    integration parameters for systematic experimentation and reproducible
    model development across different architectural variants.
    
    Args:
        config_path: Path to BERT4Rec configuration file
        
    Returns:
        Dictionary containing comprehensive training configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded BERT4Rec configuration from {config_path}")
    logging.info(f"Key parameters: max_seq_len={config['model']['max_seq_len']}, "
                f"hidden_dim={config['model']['hidden_dim']}, "
                f"num_layers={config['model']['num_layers']}")
    
    return config

class BERT4RecTrainer:
    """
    Comprehensive training orchestrator for BERT4Rec sequential recommendation.
    
    This trainer manages the complete BERT4Rec training process including masked
    item prediction, temporal sequence modeling, content feature integration,
    and evaluation against both individual performance metrics and combined
    two-stage system effectiveness.
    """
    
    def __init__(self, model: BERT4RecModel, data_loader: BERT4RecDataLoader, 
                 config: Dict[str, Any]):
        """
        Initialize trainer with model, data, and configuration parameters.
        
        Args:
            model: Initialized BERT4Rec model with content feature integration
            data_loader: Prepared data loader with user sequences and masking
            config: Complete configuration including training and evaluation parameters
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        # Initialize optimizer with appropriate learning rate for transformer training
        # BERT4Rec typically requires lower learning rates than simpler architectures
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler for stable transformer training
        # The warmup helps with training stability for deep attention models
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['training']['num_epochs']
        )
        
        # Training history tracking for analysis and visualization
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Loss function for masked item prediction
        # We ignore predictions for padded positions and mask tokens
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    def compute_masked_lm_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute masked language modeling loss for BERT4Rec training.
        
        This method implements the core learning objective that teaches BERT4Rec
        to predict masked items based on bidirectional context. The loss function
        focuses on positive user interactions, teaching the model to understand
        preference patterns through contextual prediction.
        
        Args:
            batch: Training batch with masked sequences and targets
            
        Returns:
            Loss tensor for backpropagation and optimization
        """
        # Extract batch components
        masked_sequences = batch['masked_sequence']  # [batch_size, seq_len]
        content_features = batch['content_features']  # [batch_size, seq_len, feature_dim]
        mask_positions = batch['mask_positions']  # [batch_size, max_masks]
        target_items = batch['target_items']  # [batch_size, max_masks] 
        num_masks = batch['num_masks']  # [batch_size]
        
        batch_size = masked_sequences.shape[0]
        
        # Forward pass through BERT4Rec model
        # This produces contextualized representations for each position
        sequence_output = self.model(masked_sequences, content_features)
        
        # Extract representations at masked positions for prediction
        # We only compute loss for actual masked positions, not padding
        total_loss = 0.0
        total_predictions = 0
        
        for i in range(batch_size):
            # Get number of actual masks for this sequence (excluding padding)
            n_masks = num_masks[i].item()
            if n_masks == 0:
                continue
                
            # Extract masked position representations
            valid_mask_positions = mask_positions[i][:n_masks]
            valid_targets = target_items[i][:n_masks]
            
            # Get contextualized representations at masked positions
            masked_representations = sequence_output[i][valid_mask_positions]  # [n_masks, hidden_dim]
            
            # Project to item vocabulary for prediction
            # We compute similarity with all possible item embeddings
            item_embeddings = self.model.item_embedding.weight[1:self.model.num_items+1]  # Exclude padding
            prediction_scores = torch.matmul(masked_representations, item_embeddings.T)  # [n_masks, num_items]
            
            # Compute cross-entropy loss for masked item prediction
            # Target items are already in the correct index space (1-indexed)
            valid_targets_adjusted = valid_targets - 1  # Convert to 0-indexed for loss computation
            mask_loss = self.criterion(prediction_scores, valid_targets_adjusted)
            
            total_loss += mask_loss * n_masks  # Weight by number of predictions
            total_predictions += n_masks
        
        # Return average loss across all predictions in the batch
        return total_loss / total_predictions if total_predictions > 0 else torch.tensor(0.0, device=masked_sequences.device)
    
    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Execute one complete training epoch over all BERT4Rec training data.
        
        This method processes all user sequences with masking to teach the model
        bidirectional understanding of user preferences through contextual prediction.
        The training process focuses on learning temporal patterns while integrating
        rich content features for enhanced recommendation understanding.
        
        Args:
            data_loader: PyTorch DataLoader providing masked sequence batches
            
        Returns:
            Average masked language modeling loss across all batches
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to appropriate device (GPU if available)
            batch = {k: v.to(next(self.model.parameters()).device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Clear gradients from previous iteration
            self.optimizer.zero_grad()
            
            # Compute masked language modeling loss
            loss = self.compute_masked_lm_loss(batch)
            
            # Backpropagate gradients and update parameters
            loss.backward()
            
            # Gradient clipping for stable transformer training
            # This prevents exploding gradients in deep attention networks
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update model parameters
            self.optimizer.step()
            
            # Accumulate loss for epoch tracking
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress for long training runs
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Batch {batch_idx}/{len(data_loader)}: Loss={loss.item():.4f}, LR={current_lr:.2e}")
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate_epoch(self, data_loader: DataLoader) -> float:
        """
        Evaluate BERT4Rec performance on validation data.
        
        This evaluation measures how well the model predicts masked items
        based on bidirectional context, providing insight into the quality
        of learned sequential patterns and content integration.
        
        Args:
            data_loader: Validation data loader with masked sequences
            
        Returns:
            Average validation loss across all batches
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(next(self.model.parameters()).device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Compute validation loss
                loss = self.compute_masked_lm_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self) -> Dict[str, List]:
        """
        Execute complete BERT4Rec training pipeline with temporal validation.
        
        This method orchestrates the full training process including data splitting,
        batch processing, validation monitoring, and learning rate scheduling.
        The training respects temporal boundaries to ensure realistic evaluation
        of sequential modeling capabilities.
        
        Returns:
            Complete training history with losses and metrics over time
        """
        config = self.config['training']
        
        logging.info(f"Starting BERT4Rec training for {config['num_epochs']} epochs")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create temporal train/validation split for sequences
        # This ensures we evaluate sequential modeling on future interactions
        train_sequences, val_sequences = self._create_temporal_sequence_split(
            validation_ratio=config['validation_split']
        )
        
        # Create datasets with masking strategy
        train_dataset = BERT4RecDataset(
            train_sequences,
            self.data_loader.movie_features,
            self.data_loader.movie_feature_map,
            max_seq_len=self.config['model']['max_seq_len'],
            mask_prob=self.config['training']['mask_prob'],
            num_items=self.data_loader.num_movies
        )
        
        val_dataset = BERT4RecDataset(
            val_sequences,
            self.data_loader.movie_features, 
            self.data_loader.movie_feature_map,
            max_seq_len=self.config['model']['max_seq_len'],
            mask_prob=self.config['training']['mask_prob'],
            num_items=self.data_loader.num_movies
        )
        
        # Create data loaders for efficient batch processing
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        logging.info(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)}")
        
        # Training loop with validation monitoring
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['num_epochs']):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = self.evaluate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch results
            logging.info(f"Epoch {epoch+1}/{config['num_epochs']}: "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"LR: {current_lr:.2e}")
            
            # Store training history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(current_lr)
            
            # Early stopping based on validation performance
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.get('early_stopping_patience', 10):
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logging.info("BERT4Rec training completed successfully")
        return self.training_history
    
    def _create_temporal_sequence_split(self, validation_ratio: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Create temporal split of user sequences for validation.
        
        This method implements temporal splitting at the sequence level, ensuring
        that validation sequences represent later user interactions than training
        sequences. This temporal separation enables realistic evaluation of
        sequential modeling performance.
        
        Args:
            validation_ratio: Fraction of recent sequences for validation
            
        Returns:
            Tuple of (train_sequences, val_sequences) with temporal ordering
        """
        all_sequences = []
        
        # Collect all sequences with timestamps for temporal sorting
        for user_idx, interactions in self.data_loader.user_sequences.items():
            if interactions and len(interactions) >= 3:
                # Use the timestamp of the last interaction for sequence ordering
                last_timestamp = max(interaction.get('timestamp', 0) for interaction in interactions)
                all_sequences.append({
                    'user_idx': user_idx,
                    'interactions': interactions,
                    'last_timestamp': last_timestamp
                })
        
        # Sort sequences by temporal order (earliest to latest)
        all_sequences.sort(key=lambda x: x['last_timestamp'])
        
        # Create temporal split
        split_idx = int(len(all_sequences) * (1 - validation_ratio))
        train_sequence_data = all_sequences[:split_idx]
        val_sequence_data = all_sequences[split_idx:]
        
        # Convert back to user_sequences format
        train_sequences = {seq['user_idx']: seq['interactions'] for seq in train_sequence_data}
        val_sequences = {seq['user_idx']: seq['interactions'] for seq in val_sequence_data}
        
        logging.info(f"Temporal sequence split: {len(train_sequences)} train users, {len(val_sequences)} val users")
        
        return train_sequences, val_sequences


def save_bert4rec_artifacts(model: BERT4RecModel, trainer: BERT4RecTrainer, 
                          data_loader: BERT4RecDataLoader, config: Dict[str, Any],
                          output_dir: Path) -> None:
    """
    Save all BERT4Rec artifacts for inference and integration analysis.
    
    This function creates the complete artifact set needed for BERT4Rec inference
    and integration with your two-tower retrieval system. The artifacts enable
    both standalone sequential modeling and combined two-stage recommendation.
    
    Args:
        model: Trained BERT4Rec model with learned sequential patterns
        trainer: Trainer containing training history and metrics
        data_loader: Data loader with sequence processing parameters
        config: Complete training configuration
        output_dir: Directory for artifact storage
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained model state dictionary
    model_path = output_dir / "bert4rec_model.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved BERT4Rec model weights to {model_path}")
    
    # Save model architecture and configuration for inference
    model_config = {
        'num_items': model.num_items,
        'content_feature_dim': model.content_projection.in_features,
        'max_seq_len': model.max_seq_len,
        'hidden_dim': model.hidden_dim,
        'num_layers': len(model.transformer_blocks),
        'num_heads': model.transformer_blocks[0].attention.num_heads,
        'training_config': config
    }
    
    config_path = output_dir / "bert4rec_config.json" 
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logging.info(f"Saved BERT4Rec configuration to {config_path}")
    
    # Save training history for analysis
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
    logging.info(f"Saved training history to {history_path}")
    
    # Save data processing artifacts for consistent inference
    data_artifacts = {
        'user_to_idx': data_loader.user_to_idx,
        'movie_to_idx': data_loader.movie_to_idx,
        'movie_feature_map': data_loader.movie_feature_map,
        'num_users': data_loader.num_users,
        'num_movies': data_loader.num_movies
    }
    
    data_path = output_dir / "data_artifacts.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(data_artifacts, f)
    logging.info(f"Saved data processing artifacts to {data_path}")


def main():
    """
    Main function orchestrating complete BERT4Rec training pipeline.
    
    This function implements the DVC entry point for BERT4Rec training,
    handling configuration loading, data preparation, model initialization,
    training execution, and artifact saving in a reproducible workflow.
    """
    # Parse command line arguments for DVC integration
    parser = argparse.ArgumentParser(description='Train BERT4Rec sequential recommendation model')
    parser.add_argument('--config', type=str, default='configs/bert4rec.yaml',
                       help='Path to BERT4Rec configuration file')
    args = parser.parse_args()
    
    # Initialize logging and load configuration
    setup_logging()
    config = load_config(args.config)
    
    # Set random seeds for reproducible training
    torch.manual_seed(config['training']['random_seed'])
    np.random.seed(config['training']['random_seed'])
    
    logging.info("Starting BERT4Rec training pipeline...")
    
    try:
        # Load and prepare sequential training data
        logging.info("Loading and processing sequential data...")
        data_loader = BERT4RecDataLoader(
            sequences_path=config['data']['sequences_path'],
            movies_path=config['data']['movies_path'],
            max_seq_len=config['model']['max_seq_len'],
            min_seq_len=config['data']['min_seq_len']
        )
        
        # Initialize BERT4Rec model with content feature integration
        logging.info("Initializing BERT4Rec model...")
        model = BERT4RecModel(
            num_items=data_loader.num_movies,
            content_feature_dim=data_loader.movie_features.shape[1],
            max_seq_len=config['model']['max_seq_len'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout_rate=config['model']['dropout_rate']
        )
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logging.info(f"Model moved to device: {device}")
        
        # Initialize trainer and execute training
        trainer = BERT4RecTrainer(model, data_loader, config)
        training_history = trainer.train()
        
        # Save model artifacts and training results
        output_dir = Path(config['outputs']['model_dir'])
        save_bert4rec_artifacts(model, trainer, data_loader, config, output_dir)
        
        # Save training metrics for DVC tracking
        metrics = {
            'final_train_loss': training_history['train_loss'][-1],
            'final_val_loss': training_history['val_loss'][-1],
            'num_epochs_trained': len(training_history['epoch']),
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
        
        metrics_path = Path(config['outputs']['metrics_path'])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved training metrics to {metrics_path}")
        
        logging.info("BERT4Rec training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"BERT4Rec training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()