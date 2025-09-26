"""
Two-Tower Model Training Script for DVC Pipeline

This script implements the complete training pipeline for the two-tower recommendation
model, designed to integrate seamlessly with DVC for reproducible machine learning
workflows. It handles data loading, model training, evaluation, and artifact saving
in a structured way that enables systematic experimentation and comparison.
"""

import torch
import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
import argparse
from typing import Dict, Any

# Import our two-tower components
import sys
sys.path.append('.')  # Add project root to path
from movie_genie.retrieval.two_tower_model import (
    TwoTowerModel, TwoTowerDataLoader, TwoTowerTrainer, TwoTowerEvaluator
)

def setup_logging() -> None:
    """Configure logging for the training process with detailed progress tracking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.
    
    The configuration file centralizes all hyperparameters and training settings,
    enabling systematic experimentation and ensuring that all training parameters
    are tracked by DVC for reproducibility.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing all training configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    logging.info(f"Key parameters: embedding_dim={config['model']['embedding_dim']}, "
                f"learning_rate={config['training']['learning_rate']}, "
                f"epochs={config['training']['num_epochs']}")
    
    return config

def save_model_artifacts(model: TwoTowerModel, trainer: TwoTowerTrainer,
                        config: Dict[str, Any], output_dir: Path, data_loader: 'TwoTowerDataLoader') -> None:
    """
    Save all model artifacts needed for inference and analysis.
    
    This function saves the complete set of artifacts that DVC will track,
    including the trained model weights, training configuration, and any
    derived embeddings that subsequent pipeline stages might need.
    
    Args:
        model: Trained two-tower model
        trainer: Trainer object containing training history
        config: Training configuration used
        output_dir: Directory to save all artifacts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained model state dictionary
    model_path = output_dir / "two_tower_model.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved model weights to {model_path}")
    
    # Save complete model architecture and configuration for inference
    model_config = {
        'num_users': model.num_users,
        'num_movies': model.num_movies, 
        'embedding_dim': model.embedding_dim,
        'content_feature_dim': data_loader.movie_features.shape[1],
        'training_config': config
    }
    
    config_path = output_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logging.info(f"Saved model configuration to {config_path}")
    
    # Save training history for analysis and visualization
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
    logging.info(f"Saved training history to {history_path}")

def generate_movie_embeddings(model: TwoTowerModel, data_loader: TwoTowerDataLoader,
                             output_path: Path) -> None:
    """
    Pre-compute and save movie embeddings for fast inference serving.
    
    This function generates the item embeddings that enable fast recommendation
    serving in production. By pre-computing these embeddings, the recommendation
    system can focus on user embedding computation during serving, achieving
    the sub-200ms latency targets we discussed.
    
    Args:
        model: Trained two-tower model
        data_loader: Data loader containing movie features
        output_path: Path to save the movie embeddings parquet file
    """
    logging.info("Generating pre-computed movie embeddings for serving...")
    
    model.eval()
    movie_embeddings_data = []
    
    # Process all movies to generate embeddings
    all_movie_indices = list(range(data_loader.num_movies))
    movie_tensors = torch.tensor(all_movie_indices, dtype=torch.long)
    
    with torch.no_grad():
        # Generate embeddings in batches for memory efficiency
        batch_size = 1000
        for i in range(0, len(all_movie_indices), batch_size):
            batch_indices = movie_tensors[i:i + batch_size]
            # Get movie features for this batch
            if hasattr(data_loader, 'movie_features'):
                batch_features = data_loader.movie_features[i:i + batch_size]
            else:
                # Create tensor from batch indices if movie_features not available
                batch_features = torch.zeros(len(batch_indices), model.content_feature_dim)
            
            batch_embeddings = model.get_movie_embeddings(batch_indices, batch_features)
            
            # Convert to records for DataFrame storage
            for j, embedding in enumerate(batch_embeddings):
                movie_idx = all_movie_indices[i + j]
                # Get original movie ID if mapping exists
                if hasattr(data_loader, 'idx_to_movie'):
                    original_movie_id = data_loader.idx_to_movie[movie_idx]
                else:
                    original_movie_id = movie_idx  # Use index as ID if no mapping
                
                movie_embeddings_data.append({
                    'movieId': original_movie_id,
                    'movie_idx': movie_idx,
                    'embedding': embedding.cpu().numpy().tolist()
                })
    
    # Save as parquet for efficient loading during serving
    embeddings_df = pd.DataFrame(movie_embeddings_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_parquet(output_path)
    
    logging.info(f"Saved {len(embeddings_df)} movie embeddings to {output_path}")

def load_existing_model(model_dir: Path, data_loader: TwoTowerDataLoader) -> TwoTowerModel:
    """
    Load an existing trained model if available.

    Args:
        model_dir: Directory containing saved model artifacts
        data_loader: Data loader for model architecture parameters

    Returns:
        Loaded model if successful, None otherwise
    """
    model_path = model_dir / "two_tower_model.pth"
    config_path = model_dir / "model_config.json"

    if not (model_path.exists() and config_path.exists()):
        logging.info("No existing model found - will train new model")
        return None

    try:
        # Load model configuration
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        logging.info(f"Found existing model at {model_path}")

        # Verify model configuration matches current data
        if (model_config['num_users'] != data_loader.num_users or
            model_config['num_movies'] != data_loader.num_movies or
            model_config['content_feature_dim'] != data_loader.movie_features.shape[1]):

            logging.warning("Existing model configuration doesn't match current data - will retrain")
            logging.warning(f"  Existing: {model_config['num_users']} users, {model_config['num_movies']} movies, {model_config['content_feature_dim']} features")
            logging.warning(f"  Current:  {data_loader.num_users} users, {data_loader.num_movies} movies, {data_loader.movie_features.shape[1]} features")
            return None

        # Create model with saved configuration
        training_config = model_config['training_config']
        model = TwoTowerModel(
            num_users=model_config['num_users'],
            num_movies=model_config['num_movies'],
            content_feature_dim=model_config['content_feature_dim'],
            embedding_dim=training_config['model']['embedding_dim'],
            user_hidden_dims=training_config['model']['user_hidden_dims'],
            item_hidden_dims=training_config['model']['item_hidden_dims'],
            dropout_rate=training_config['model']['dropout_rate']
        )

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logging.info("Successfully loaded existing trained model")

        return model

    except Exception as e:
        logging.warning(f"Failed to load existing model: {e}")
        logging.info("Will train new model instead")
        return None

def main():
    """
    Main training function that orchestrates the complete two-tower training pipeline.

    This function implements the entry point for DVC, handling argument parsing,
    configuration loading, data preparation, model training, evaluation, and
    artifact saving in a structured workflow that integrates with the DVC
    dependency tracking system. It includes checkpoint loading to skip training
    if a valid model already exists.
    """
    # Parse command line arguments for DVC integration
    parser = argparse.ArgumentParser(description='Train two-tower recommendation model')
    parser.add_argument('--config', type=str, default='configs/two_tower.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if existing model is found')
    args = parser.parse_args()
    
    # Setup logging and load configuration
    setup_logging()
    config = load_config(args.config)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['random_seed'])
    np.random.seed(config['training']['random_seed'])
    
    logging.info("Starting two-tower model training pipeline...")
    
    try:
        # Load and prepare data using existing data loader
        logging.info("Loading and preparing training data...")
        data_loader = TwoTowerDataLoader(
            sequences_path=config['data']['sequences_path'],
            movies_path=config['data']['movies_path'],
            negative_sampling_ratio=config['data']['negative_sampling_ratio'],
            min_user_interactions=config['data']['min_user_interactions']
        )
        
        # Check for existing trained model (unless forced to retrain)
        model_output_dir = Path(config['outputs']['model_dir'])
        model = None

        if not args.force_retrain:
            model = load_existing_model(model_output_dir, data_loader)

        if model is None:
            # Initialize new model with configuration parameters
            logging.info("Initializing new two-tower model...")
            model = TwoTowerModel(
                num_users=data_loader.num_users,
                num_movies=data_loader.num_movies,
                content_feature_dim=data_loader.movie_features.shape[1],
                embedding_dim=config['model']['embedding_dim'],
                user_hidden_dims=config['model']['user_hidden_dims'],
                item_hidden_dims=config['model']['item_hidden_dims'],
                dropout_rate=config['model']['dropout_rate']
            )

            logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

            # Initialize trainer and execute training process
            logging.info("Starting model training...")
            trainer = TwoTowerTrainer(
                model=model,
                data_loader=data_loader,
                learning_rate=config['training']['learning_rate'],
                margin=config['training']['margin']
            )

            # Execute training with configuration parameters
            training_history = trainer.train(
                num_epochs=config['training']['num_epochs'],
                batch_size=config['training']['batch_size'],
                validation_split=config['training']['validation_split']
            )

            # Save model artifacts after training
            save_model_artifacts(model, trainer, config, model_output_dir, data_loader)

        else:
            # Model loaded from checkpoint - skip training
            logging.info("Using existing trained model - skipping training")
            # Create a dummy trainer for evaluation
            trainer = TwoTowerTrainer(
                model=model,
                data_loader=data_loader,
                learning_rate=config['training']['learning_rate'],
                margin=config['training']['margin']
            )
            # Create dummy history for consistency
            training_history = {'epoch': [], 'loss': [], 'metrics': []}
        
        # Evaluate trained model performance
        logging.info("Evaluating trained model...")
        evaluator = TwoTowerEvaluator(model, data_loader)
        
        # Create test split for evaluation (in practice, you'd use held-out test data)
        test_examples = {
            'positive': data_loader.positive_examples[-1000:],  # Last 1000 positive examples
            'negative': data_loader.negative_examples[-1000:]   # Last 1000 negative examples
        }
        
        evaluation_results = evaluator.evaluate_model_performance(
            test_examples, k_values=config['evaluation']['k_values']
        )
        
        # Save evaluation metrics for DVC tracking
        metrics_path = Path(config['outputs']['metrics_path'])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        logging.info(f"Saved evaluation metrics to {metrics_path}")
        
        # Generate and save movie embeddings for serving (always regenerate for freshness)
        embeddings_path = Path(config['outputs']['embeddings_path'])
        generate_movie_embeddings(model, data_loader, embeddings_path)
        
        logging.info("Two-tower training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()