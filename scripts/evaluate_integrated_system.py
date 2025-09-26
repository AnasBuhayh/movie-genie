"""
Integrated Two-Stage Recommendation System Evaluation

This script provides comprehensive evaluation of the complete recommendation system,
combining two-tower retrieval with BERT4Rec sequential ranking. The evaluation
measures both individual component performance and integrated system effectiveness,
providing insights into how the two-stage architecture performs compared to
individual models and simpler baseline approaches.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import both model architectures
import sys
sys.path.append('.')
from movie_genie.retrieval.two_tower_model import TwoTowerModel, TwoTowerDataLoader
from movie_genie.ranking.bert4rec_model import BERT4RecModel, BERT4RecDataLoader

def setup_logging() -> None:
    """Configure detailed logging for integrated system evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

class IntegratedSystemEvaluator:
    """
    Comprehensive evaluator for the integrated two-stage recommendation system.
    
    This evaluator loads both trained models and orchestrates the complete
    recommendation pipeline: candidate generation through two-tower retrieval
    followed by sequential ranking through BERT4Rec. The evaluation measures
    both individual component quality and combined system effectiveness.
    
    The key insight involves understanding how the two stages complement each
    other. The two-tower model provides computational efficiency by quickly
    identifying potentially relevant candidates, while BERT4Rec provides
    sophisticated preference understanding by optimally ordering those candidates
    based on temporal user behavior patterns and content features.
    """
    
    def __init__(self, 
                 two_tower_model_dir: str,
                 bert4rec_model_dir: str,
                 data_config: Dict[str, Any]):
        """
        Initialize the integrated evaluator by loading both trained models.
        
        This initialization process demonstrates how production recommendation
        systems coordinate multiple model components. Each model loads with its
        specific configuration and data processing requirements, but they must
        work together seamlessly during recommendation generation.
        
        Args:
            two_tower_model_dir: Directory containing trained two-tower model artifacts
            bert4rec_model_dir: Directory containing trained BERT4Rec model artifacts  
            data_config: Configuration specifying data paths and processing parameters
        """
        self.data_config = data_config
        
        # Load both trained models and their configurations
        # This process shows how to restore complex model architectures from saved artifacts
        self.two_tower_model, self.two_tower_data_loader = self._load_two_tower_model(two_tower_model_dir)
        self.bert4rec_model, self.bert4rec_data_loader = self._load_bert4rec_model(bert4rec_model_dir)
        
        # Move models to appropriate device for evaluation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.two_tower_model = self.two_tower_model.to(self.device)
        self.bert4rec_model = self.bert4rec_model.to(self.device)
        
        # Set models to evaluation mode for consistent inference behavior
        self.two_tower_model.eval()
        self.bert4rec_model.eval()
        
        logging.info(f"Initialized integrated evaluator on device: {self.device}")
        logging.info(f"Two-tower model: {sum(p.numel() for p in self.two_tower_model.parameters()):,} parameters")
        logging.info(f"BERT4Rec model: {sum(p.numel() for p in self.bert4rec_model.parameters()):,} parameters")
    
    def _load_two_tower_model(self, model_dir: str) -> Tuple[TwoTowerModel, TwoTowerDataLoader]:
        """
        Load trained two-tower model with all necessary artifacts.
        
        This method demonstrates the process of restoring a trained model for
        inference, including loading the model architecture, trained weights,
        and data processing components needed for consistent feature preparation.
        
        Args:
            model_dir: Directory containing two-tower model artifacts
            
        Returns:
            Tuple of (loaded_model, data_loader) ready for inference
        """
        model_dir = Path(model_dir)
        
        # Load model configuration to reconstruct architecture
        with open(model_dir / "model_config.json", 'r') as f:
            config = json.load(f)
        
        logging.info("Loading two-tower model components...")
        
        # Initialize data loader with same parameters as training
        # This ensures consistent feature processing between training and inference
        data_loader = TwoTowerDataLoader(
            sequences_path=self.data_config['sequences_path'],
            movies_path=self.data_config['movies_path'],
            negative_sampling_ratio=4,  # Not used during inference but needed for initialization
            min_user_interactions=1     # Lower threshold for evaluation
        )
        
        # Reconstruct model architecture using saved configuration
        model = TwoTowerModel(
            num_users=config['num_users'],
            num_movies=config['num_movies'],
            content_feature_dim=config['content_feature_dim'],
            embedding_dim=config['embedding_dim']
        )
        
        # Load trained weights
        model_weights_path = model_dir / "two_tower_model.pth"
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        
        logging.info(f"Loaded two-tower model: {config['num_users']} users, {config['num_movies']} movies")
        
        return model, data_loader
    
    def _load_bert4rec_model(self, model_dir: str) -> Tuple[BERT4RecModel, BERT4RecDataLoader]:
        """
        Load trained BERT4Rec model with sequence processing capabilities.
        
        Loading BERT4Rec requires careful attention to the sequence processing
        parameters and content feature integration that enable the model to
        understand temporal user preferences and content characteristics.
        
        Args:
            model_dir: Directory containing BERT4Rec model artifacts
            
        Returns:
            Tuple of (loaded_model, data_loader) configured for sequential ranking
        """
        model_dir = Path(model_dir)
        
        # Load model configuration for architecture reconstruction
        with open(model_dir / "bert4rec_config.json", 'r') as f:
            config = json.load(f)
        
        logging.info("Loading BERT4Rec model components...")
        
        # Initialize data loader with sequence processing capabilities
        # The data loader handles conversion from user interaction history to
        # the sequence format that BERT4Rec expects for ranking
        data_loader = BERT4RecDataLoader(
            sequences_path=self.data_config['sequences_path'],
            movies_path=self.data_config['movies_path'],
            max_seq_len=config['max_seq_len'],
            min_seq_len=3
        )
        
        # Reconstruct BERT4Rec architecture
        model = BERT4RecModel(
            num_items=config['num_items'],
            content_feature_dim=config['content_feature_dim'],
            max_seq_len=config['max_seq_len'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads']
        )
        
        # Load trained transformer weights
        model_weights_path = model_dir / "bert4rec_model.pth"
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        
        # Load data processing artifacts for consistent ID mapping
        with open(model_dir / "data_artifacts.pkl", 'rb') as f:
            data_artifacts = pickle.load(f)
        
        # Ensure consistent mappings between models
        # This alignment is crucial for the two-stage pipeline to work correctly
        self._verify_id_mappings(data_artifacts)
        
        logging.info(f"Loaded BERT4Rec model: {config['max_seq_len']} max sequence length, "
                    f"{config['hidden_dim']} hidden dimensions")
        
        return model, data_loader
    
    def _verify_id_mappings(self, bert4rec_data_artifacts: Dict[str, Any]) -> None:
        """
        Verify consistent ID mappings between two-tower and BERT4Rec models.
        
        This verification ensures that both models use the same user and movie
        ID mappings, which is essential for the two-stage pipeline to function
        correctly. Inconsistent mappings would cause the candidate items from
        the two-tower model to be misinterpreted by BERT4Rec.
        
        Args:
            bert4rec_data_artifacts: Data processing artifacts from BERT4Rec training
        """
        # Compare user and movie mappings between models
        two_tower_users = set(self.two_tower_data_loader.user_to_idx.keys())
        bert4rec_users = set(bert4rec_data_artifacts['user_to_idx'].keys())
        user_overlap = len(two_tower_users.intersection(bert4rec_users))
        
        two_tower_movies = set(self.two_tower_data_loader.movie_to_idx.keys())
        bert4rec_movies = set(bert4rec_data_artifacts['movie_to_idx'].keys())
        movie_overlap = len(two_tower_movies.intersection(bert4rec_movies))
        
        logging.info(f"ID mapping verification:")
        logging.info(f"  User overlap: {user_overlap}/{len(two_tower_users)} ({user_overlap/len(two_tower_users)*100:.1f}%)")
        logging.info(f"  Movie overlap: {movie_overlap}/{len(two_tower_movies)} ({movie_overlap/len(two_tower_movies)*100:.1f}%)")
        
        if user_overlap < len(two_tower_users) * 0.9 or movie_overlap < len(two_tower_movies) * 0.9:
            logging.warning("Significant ID mapping mismatch detected between models")
    
    def generate_two_tower_candidates(self, user_idx: int, k: int = 100) -> Tuple[List[int], List[float]]:
        """
        Generate candidate recommendations using the two-tower retrieval model.
        
        This method demonstrates the first stage of your recommendation pipeline.
        The two-tower model efficiently scans your entire movie catalog to identify
        candidates that might interest the user, based on learned user and item
        embeddings that capture collaborative filtering patterns and content features.
        
        Args:
            user_idx: Index of user for candidate generation
            k: Number of candidates to retrieve
            
        Returns:
            Tuple of (candidate_movie_indices, similarity_scores) ranked by relevance
        """
        with torch.no_grad():
            # Generate user embedding using the trained user tower
            # This embedding captures the user's learned preference patterns
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            user_embedding = self.two_tower_model.get_user_embeddings(user_tensor)
            
            # Get all movie embeddings for similarity computation
            # In production, these would be pre-computed and cached for efficiency
            all_movie_indices = list(range(self.two_tower_data_loader.num_movies))
            movie_tensors = torch.tensor(all_movie_indices, dtype=torch.long, device=self.device)
            
            # Ensure movie features are on the correct device
            movie_features = self.two_tower_data_loader.movie_features.to(self.device)
            movie_embeddings = self.two_tower_model.get_movie_embeddings(movie_tensors, movie_features)
            
            # Compute similarity scores between user and all movies
            # The dot product gives cosine similarity since embeddings are normalized
            similarity_scores = torch.matmul(user_embedding, movie_embeddings.T).squeeze()
            
            # Get top-k most similar movies as candidates
            top_k_values, top_k_indices = torch.topk(similarity_scores, k=min(k, len(all_movie_indices)))
            
            # Convert to lists for easier handling in subsequent processing
            candidate_indices = top_k_indices.cpu().numpy().tolist()
            candidate_scores = top_k_values.cpu().numpy().tolist()
            
        return candidate_indices, candidate_scores
    
    def rank_candidates_with_bert4rec(self, user_idx: int, candidate_indices: List[int], 
                                    user_sequence: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        """
        Rank candidate movies using BERT4Rec sequential modeling.
        
        This method implements the second stage of your recommendation pipeline.
        BERT4Rec uses the user's interaction sequence and rich content features
        to understand temporal preference patterns and optimally order the
        candidates provided by the two-tower retrieval system.
        
        The ranking process considers both the sequential context of user preferences
        and the content characteristics of candidate movies, enabling sophisticated
        understanding of user intent and preference evolution.
        
        Args:
            user_idx: Index of user for ranking
            candidate_indices: Movie indices from two-tower candidate generation
            user_sequence: User's interaction history for sequential context
            
        Returns:
            List of (movie_index, ranking_score) tuples ordered by predicted preference
        """
        if not user_sequence or len(candidate_indices) == 0:
            # Handle edge cases where ranking is not possible
            return [(idx, 0.0) for idx in candidate_indices]
        
        with torch.no_grad():
            # Prepare user sequence for BERT4Rec input
            # This involves converting interaction history to the format expected by the transformer
            sequence_items = [interaction['movie_idx'] for interaction in user_sequence]
            
            # Truncate or pad sequence to model's expected length
            max_len = self.bert4rec_model.max_seq_len
            if len(sequence_items) > max_len:
                sequence_items = sequence_items[-max_len:]  # Take most recent interactions
            
            # Pad sequence if necessary (though typically not needed for ranking)
            padded_sequence = sequence_items + [0] * (max_len - len(sequence_items))
            
            # Get content features for sequence items
            sequence_features = []
            for item_idx in padded_sequence:
                if item_idx == 0:  # Padding token
                    sequence_features.append(torch.zeros(self.bert4rec_data_loader.movie_features.shape[1]))
                else:
                    feature_idx = self.bert4rec_data_loader.movie_feature_map.get(item_idx, 0)
                    sequence_features.append(self.bert4rec_data_loader.movie_features[feature_idx])
            
            sequence_features = torch.stack(sequence_features).unsqueeze(0).to(self.device)  # Add batch dimension
            sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long, device=self.device)
            
            # Get contextualized sequence representation from BERT4Rec
            # This representation captures the user's current preference state based on their history
            sequence_output = self.bert4rec_model(sequence_tensor, sequence_features)
            
            # Use the representation from the last non-padding position for candidate scoring
            last_position = min(len(sequence_items) - 1, max_len - 1)
            user_context_embedding = sequence_output[0, last_position]  # [hidden_dim]
            
            # Get embeddings for candidate movies
            candidate_tensors = torch.tensor(candidate_indices, dtype=torch.long, device=self.device)
            candidate_features = []
            
            for candidate_idx in candidate_indices:
                feature_idx = self.bert4rec_data_loader.movie_feature_map.get(candidate_idx, 0)
                candidate_features.append(self.bert4rec_data_loader.movie_features[feature_idx])
            
            candidate_features = torch.stack(candidate_features).to(self.device)
            
            # Get candidate embeddings (item embedding + content features)
            candidate_item_embeddings = self.bert4rec_model.item_embedding(candidate_tensors)
            candidate_content_embeddings = self.bert4rec_model.content_projection(candidate_features)
            candidate_embeddings = candidate_item_embeddings + candidate_content_embeddings
            
            # Compute ranking scores based on similarity with user context
            # Higher scores indicate better fit with the user's current preference trajectory
            ranking_scores = torch.matmul(user_context_embedding.unsqueeze(0), candidate_embeddings.T)
            ranking_scores = ranking_scores.squeeze(0).cpu().numpy()
            
            # Create ranked list of candidates with scores
            candidate_scores = list(zip(candidate_indices, ranking_scores))
            candidate_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
            
        return candidate_scores
    
    def generate_integrated_recommendations(self, user_idx: int, 
                                          candidate_k: int = 100, 
                                          final_k: int = 20) -> Dict[str, Any]:
        """
        Generate complete recommendations using the integrated two-stage system.
        
        This method orchestrates the complete recommendation pipeline, demonstrating
        how the two models work together to provide high-quality personalized
        recommendations. The process shows the complementary strengths of each stage:
        efficient candidate generation followed by sophisticated ranking.
        
        Args:
            user_idx: Index of user for recommendation generation
            candidate_k: Number of candidates to retrieve from two-tower model
            final_k: Number of final recommendations to return after ranking
            
        Returns:
            Dictionary containing recommendations, intermediate results, and analysis
        """
        results = {'user_idx': user_idx}
        
        # Stage 1: Two-Tower Candidate Generation
        # This stage efficiently scans the full catalog to identify potentially relevant items
        candidate_indices, candidate_scores = self.generate_two_tower_candidates(user_idx, candidate_k)
        results['candidates'] = {
            'indices': candidate_indices,
            'scores': candidate_scores,
            'count': len(candidate_indices)
        }
        
        # Get user's interaction sequence for BERT4Rec context
        # The sequence provides temporal context that enables sophisticated preference understanding
        if user_idx in self.bert4rec_data_loader.user_sequences:
            user_sequence = self.bert4rec_data_loader.user_sequences[user_idx]
        else:
            user_sequence = []
            logging.warning(f"No sequence found for user {user_idx}, using empty sequence")
        
        results['user_sequence_length'] = len(user_sequence)
        
        # Stage 2: BERT4Rec Sequential Ranking
        # This stage uses temporal preference patterns to optimally order the candidates
        if candidate_indices and user_sequence:
            ranked_candidates = self.rank_candidates_with_bert4rec(user_idx, candidate_indices, user_sequence)
            
            # Extract final recommendations
            final_recommendations = ranked_candidates[:final_k]
            results['recommendations'] = {
                'indices': [idx for idx, score in final_recommendations],
                'scores': [score for idx, score in final_recommendations],
                'count': len(final_recommendations)
            }
            
            # Analyze ranking changes between two-tower and BERT4Rec
            # This analysis shows how sequential modeling affects recommendation ordering
            results['ranking_analysis'] = self._analyze_ranking_changes(
                candidate_indices, candidate_scores, ranked_candidates
            )
        else:
            # Fallback to two-tower results if BERT4Rec ranking is not possible
            results['recommendations'] = {
                'indices': candidate_indices[:final_k],
                'scores': candidate_scores[:final_k],
                'count': min(final_k, len(candidate_indices))
            }
            results['ranking_analysis'] = {'fallback_used': True}
        
        return results
    
    def _analyze_ranking_changes(self, original_indices: List[int], original_scores: List[float],
                               reranked_results: List[Tuple[int, float]]) -> Dict[str, Any]:
        """
        Analyze how BERT4Rec reranking changes the order from two-tower candidates.
        
        This analysis provides insights into how sequential modeling affects recommendation
        quality. Significant reordering suggests that temporal preference patterns provide
        important signals that pure collaborative filtering misses.
        
        Args:
            original_indices: Movie indices from two-tower ranking
            original_scores: Similarity scores from two-tower model
            reranked_results: (movie_index, ranking_score) tuples from BERT4Rec
            
        Returns:
            Dictionary containing ranking change analysis and insights
        """
        # Create mapping from movie index to original rank
        original_ranks = {idx: rank for rank, idx in enumerate(original_indices)}
        
        # Calculate rank changes for reranked items
        rank_changes = []
        reranked_indices = [idx for idx, score in reranked_results]
        
        for new_rank, movie_idx in enumerate(reranked_indices):
            if movie_idx in original_ranks:
                original_rank = original_ranks[movie_idx]
                rank_change = original_rank - new_rank  # Positive means moved up, negative means moved down
                rank_changes.append({
                    'movie_idx': movie_idx,
                    'original_rank': original_rank,
                    'new_rank': new_rank,
                    'rank_change': rank_change
                })
        
        # Compute summary statistics
        if rank_changes:
            rank_change_values = [change['rank_change'] for change in rank_changes]
            analysis = {
                'total_items': len(rank_changes),
                'avg_rank_change': np.mean(rank_change_values),
                'std_rank_change': np.std(rank_change_values),
                'max_improvement': max(rank_change_values) if rank_change_values else 0,
                'max_decline': min(rank_change_values) if rank_change_values else 0,
                'items_improved': sum(1 for change in rank_change_values if change > 0),
                'items_declined': sum(1 for change in rank_change_values if change < 0),
                'rank_correlation': self._compute_rank_correlation(original_indices, reranked_indices)
            }
        else:
            analysis = {'total_items': 0, 'no_overlap': True}
        
        return analysis
    
    def _compute_rank_correlation(self, original_order: List[int], reranked_order: List[int]) -> float:
        """Compute Spearman rank correlation between original and reranked orders."""
        # Find common items between the two rankings
        common_items = set(original_order).intersection(set(reranked_order))
        
        if len(common_items) < 2:
            return 0.0
        
        # Get ranks for common items in both orderings
        original_ranks = []
        reranked_ranks = []
        
        for item in common_items:
            original_ranks.append(original_order.index(item))
            reranked_ranks.append(reranked_order.index(item))
        
        # Compute Spearman correlation
        correlation_matrix = np.corrcoef(original_ranks, reranked_ranks)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
    
    def evaluate_system_performance(self, test_users: List[int], 
                                   candidate_k_values: List[int] = [50, 100, 200],
                                   final_k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of integrated system performance across multiple metrics.
        
        This evaluation measures both the individual component performance and the
        effectiveness of the complete two-stage system. The analysis provides insights
        into how different candidate set sizes and ranking depths affect recommendation
        quality and computational efficiency.
        
        Args:
            test_users: List of user indices for evaluation
            candidate_k_values: Different candidate set sizes to evaluate
            final_k_values: Different final recommendation list sizes to evaluate
            
        Returns:
            Comprehensive evaluation results with performance metrics and analysis
        """
        logging.info(f"Starting comprehensive system evaluation on {len(test_users)} test users")
        
        evaluation_results = {
            'test_users_count': len(test_users),
            'candidate_k_values': candidate_k_values,
            'final_k_values': final_k_values,
            'individual_results': [],
            'aggregate_metrics': {}
        }
        
        # Collect individual user results for detailed analysis
        user_results = []
        
        for i, user_idx in enumerate(test_users):
            if i % 100 == 0:
                logging.info(f"Evaluating user {i+1}/{len(test_users)}")
            
            # Generate recommendations with different parameter combinations
            user_result = {'user_idx': user_idx, 'results_by_config': {}}
            
            for candidate_k in candidate_k_values:
                for final_k in final_k_values:
                    config_key = f"c{candidate_k}_f{final_k}"
                    
                    try:
                        recommendations = self.generate_integrated_recommendations(
                            user_idx, candidate_k, final_k
                        )
                        user_result['results_by_config'][config_key] = recommendations
                        
                    except Exception as e:
                        logging.warning(f"Failed to generate recommendations for user {user_idx}, "
                                      f"config {config_key}: {e}")
                        user_result['results_by_config'][config_key] = {'error': str(e)}
            
            user_results.append(user_result)
        
        evaluation_results['individual_results'] = user_results
        
        # Compute aggregate metrics across all users and configurations
        aggregate_metrics = self._compute_aggregate_metrics(user_results, candidate_k_values, final_k_values)
        evaluation_results['aggregate_metrics'] = aggregate_metrics
        
        # Analyze system characteristics and patterns
        system_analysis = self._analyze_system_characteristics(user_results)
        evaluation_results['system_analysis'] = system_analysis
        
        logging.info("Integrated system evaluation completed")
        return evaluation_results
    
    def _compute_aggregate_metrics(self, user_results: List[Dict], 
                                 candidate_k_values: List[int], 
                                 final_k_values: List[int]) -> Dict[str, Any]:
        """
        Compute aggregate performance metrics across all users and configurations.
        
        This analysis provides system-level insights into how different parameter
        choices affect recommendation quality, computational efficiency, and user
        coverage. The metrics help guide decisions about optimal configurations
        for production deployment.
        
        Args:
            user_results: Individual user evaluation results
            candidate_k_values: Candidate set sizes evaluated
            final_k_values: Final recommendation sizes evaluated
            
        Returns:
            Dictionary containing aggregate metrics and performance analysis
        """
        metrics = {}
        
        # Analyze performance across different configuration combinations
        for candidate_k in candidate_k_values:
            for final_k in final_k_values:
                config_key = f"c{candidate_k}_f{final_k}"
                
                # Collect metrics for this configuration
                successful_users = 0
                total_candidates = []
                total_recommendations = []
                ranking_correlations = []
                rank_improvements = []
                
                for user_result in user_results:
                    if config_key in user_result['results_by_config']:
                        result = user_result['results_by_config'][config_key]
                        
                        if 'error' not in result:
                            successful_users += 1
                            
                            # Candidate generation metrics
                            if 'candidates' in result:
                                total_candidates.append(result['candidates']['count'])
                            
                            # Final recommendation metrics
                            if 'recommendations' in result:
                                total_recommendations.append(result['recommendations']['count'])
                            
                            # Ranking analysis metrics
                            if 'ranking_analysis' in result and 'rank_correlation' in result['ranking_analysis']:
                                ranking_correlations.append(result['ranking_analysis']['rank_correlation'])
                            
                            if 'ranking_analysis' in result and 'avg_rank_change' in result['ranking_analysis']:
                                rank_improvements.append(result['ranking_analysis']['avg_rank_change'])
                
                # Compute aggregate statistics for this configuration
                config_metrics = {
                    'successful_users': successful_users,
                    'success_rate': successful_users / len(user_results) if user_results else 0,
                    'avg_candidates': np.mean(total_candidates) if total_candidates else 0,
                    'avg_recommendations': np.mean(total_recommendations) if total_recommendations else 0,
                    'avg_rank_correlation': np.mean(ranking_correlations) if ranking_correlations else 0,
                    'avg_rank_improvement': np.mean(rank_improvements) if rank_improvements else 0
                }
                
                metrics[config_key] = config_metrics
        
        return metrics
    
    def _analyze_system_characteristics(self, user_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze system-level characteristics and patterns in recommendation generation.
        
        This analysis provides insights into how the integrated system behaves
        across different types of users and scenarios. Understanding these patterns
        helps identify opportunities for improvement and potential deployment challenges.
        
        Args:
            user_results: Individual user evaluation results
            
        Returns:
            Dictionary containing system characteristic analysis
        """
        analysis = {
            'user_coverage': {},
            'sequence_length_effects': {},
            'recommendation_diversity': {},
            'performance_patterns': {}
        }
        
        # Analyze user coverage and success patterns
        users_with_sequences = 0
        users_with_recommendations = 0
        sequence_lengths = []
        
        for user_result in user_results:
            # Check if user has interaction sequence
            sample_result = next(iter(user_result['results_by_config'].values()))
            if 'user_sequence_length' in sample_result:
                seq_len = sample_result['user_sequence_length']
                if seq_len > 0:
                    users_with_sequences += 1
                    sequence_lengths.append(seq_len)
            
            # Check if user received recommendations
            has_recommendations = any(
                'recommendations' in result and result['recommendations']['count'] > 0
                for result in user_result['results_by_config'].values()
                if 'error' not in result
            )
            if has_recommendations:
                users_with_recommendations += 1
        
        analysis['user_coverage'] = {
            'total_users': len(user_results),
            'users_with_sequences': users_with_sequences,
            'users_with_recommendations': users_with_recommendations,
            'sequence_coverage_rate': users_with_sequences / len(user_results) if user_results else 0,
            'recommendation_coverage_rate': users_with_recommendations / len(user_results) if user_results else 0
        }
        
        # Analyze sequence length effects on recommendation quality
        if sequence_lengths:
            analysis['sequence_length_effects'] = {
                'min_length': min(sequence_lengths),
                'max_length': max(sequence_lengths),
                'mean_length': np.mean(sequence_lengths),
                'median_length': np.median(sequence_lengths),
                'std_length': np.std(sequence_lengths)
            }
        
        return analysis


def save_evaluation_results(results: Dict[str, Any], output_path: Path) -> None:
    """
    Save comprehensive evaluation results for analysis and reporting.
    
    This function organizes the evaluation results into structured formats
    suitable for further analysis, visualization, and reporting to stakeholders.
    The saved artifacts enable detailed investigation of system performance
    and comparison with alternative approaches.
    
    Args:
        results: Complete evaluation results from integrated system assessment
        output_path: Path for saving evaluation artifacts
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main results as JSON for easy loading and analysis
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Saved evaluation results to {output_path}")
    
    # Extract and save key metrics for DVC tracking
    if 'aggregate_metrics' in results:
        key_metrics = {}
        for config, metrics in results['aggregate_metrics'].items():
            key_metrics[f"{config}_success_rate"] = metrics['success_rate']
            key_metrics[f"{config}_avg_rank_correlation"] = metrics['avg_rank_correlation']
            key_metrics[f"{config}_avg_rank_improvement"] = metrics['avg_rank_improvement']
        
        metrics_path = output_path.parent / "key_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(key_metrics, f, indent=2)
        
        logging.info(f"Saved key metrics to {metrics_path}")


def main():
    """
    Main function orchestrating comprehensive integrated system evaluation.
    
    This function demonstrates how to evaluate a complete two-stage recommendation
    system, measuring both individual component performance and integrated system
    effectiveness. The evaluation provides actionable insights for system optimization
    and deployment planning.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate integrated recommendation system')
    parser.add_argument('--two_tower_dir', type=str, default='models/two_tower',
                       help='Directory containing trained two-tower model')
    parser.add_argument('--bert4rec_dir', type=str, default='models/bert4rec',
                       help='Directory containing trained BERT4Rec model')
    parser.add_argument('--sequences_path', type=str, 
                       default='data/processed/sequences_with_metadata.parquet',
                       help='Path to user interaction sequences')
    parser.add_argument('--movies_path', type=str,
                       default='data/processed/movies_with_content_features.parquet', 
                       help='Path to movie content features')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory for evaluation results')
    parser.add_argument('--num_test_users', type=int, default=500,
                       help='Number of users to include in evaluation')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("Starting integrated recommendation system evaluation")
    
    try:
        # Initialize integrated evaluator
        data_config = {
            'sequences_path': args.sequences_path,
            'movies_path': args.movies_path
        }
        
        evaluator = IntegratedSystemEvaluator(
            args.two_tower_dir,
            args.bert4rec_dir, 
            data_config
        )
        
        # Select test users for evaluation
        # In practice, you would use a held-out test set with temporal splitting
        all_users = list(evaluator.bert4rec_data_loader.user_sequences.keys())
        test_users = np.random.choice(all_users, 
                                    size=min(args.num_test_users, len(all_users)), 
                                    replace=False).tolist()
        
        logging.info(f"Evaluating system on {len(test_users)} test users")
        
        # Execute comprehensive evaluation
        evaluation_results = evaluator.evaluate_system_performance(
            test_users=test_users,
            candidate_k_values=[50, 100, 200],
            final_k_values=[5, 10, 20]
        )
        
        # Save evaluation results
        output_path = Path(args.output_dir) / "integrated_system_evaluation.json"
        save_evaluation_results(evaluation_results, output_path)
        
        # Log key findings
        aggregate_metrics = evaluation_results.get('aggregate_metrics', {})
        if aggregate_metrics:
            logging.info("Key evaluation findings:")
            for config, metrics in aggregate_metrics.items():
                logging.info(f"  {config}: Success rate {metrics['success_rate']:.3f}, "
                           f"Rank correlation {metrics['avg_rank_correlation']:.3f}")
        
        logging.info("Integrated system evaluation completed successfully")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()