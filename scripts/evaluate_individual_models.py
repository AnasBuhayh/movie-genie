#!/usr/bin/env python3
"""
Comprehensive evaluation of individual recommendation models.

This script evaluates Two-Tower and BERT4Rec models separately using standard
recommendation metrics: Recall@K, Precision@K, nDCG@K, MRR, Coverage, Diversity.
"""

import torch
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import math
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append('.')

from movie_genie.retrieval.two_tower_model import TwoTowerDataLoader, TwoTowerModel
from movie_genie.ranking.bert4rec_model import BERT4RecDataLoader, BERT4RecModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RecommendationEvaluator:
    """Comprehensive evaluator for recommendation models."""

    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        self.k_values = k_values
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def recall_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_items:
            return 0.0

        recommended_k = set(recommended_items[:k])
        hits = len(recommended_k & relevant_items)
        return hits / len(relevant_items)

    def precision_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0

        recommended_k = set(recommended_items[:k])
        hits = len(recommended_k & relevant_items)
        return hits / k

    def ndcg_at_k(self, recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate nDCG@K."""
        def dcg(items, relevant_set, k):
            score = 0.0
            for i, item in enumerate(items[:k]):
                if item in relevant_set:
                    score += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
            return score

        if not relevant_items:
            return 0.0

        # DCG of recommendations
        dcg_score = dcg(recommended_items, relevant_items, k)

        # IDCG (best possible DCG)
        ideal_items = list(relevant_items) + [0] * k  # Pad with irrelevant items
        idcg_score = dcg(ideal_items, relevant_items, k)

        return dcg_score / idcg_score if idcg_score > 0 else 0.0

    def mrr(self, recommended_items: List[int], relevant_items: Set[int]) -> float:
        """Calculate Mean Reciprocal Rank for a single user."""
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0

    def catalog_coverage(self, all_recommendations: List[List[int]], total_items: int) -> float:
        """Calculate what fraction of the catalog gets recommended."""
        recommended_items = set()
        for user_recs in all_recommendations:
            recommended_items.update(user_recs)
        return len(recommended_items) / total_items

    def diversity_metrics(self, recommendations: List[List[int]]) -> Dict[str, float]:
        """Calculate diversity metrics."""
        # Intra-list diversity (avg pairwise distance within each user's recommendations)
        # Inter-list diversity (how different are recommendations across users)

        if not recommendations:
            return {'intra_diversity': 0.0, 'inter_diversity': 0.0, 'gini_coefficient': 0.0}

        # Item popularity distribution (for Gini coefficient)
        item_counts = defaultdict(int)
        total_recs = 0

        for user_recs in recommendations:
            for item in user_recs[:20]:  # Consider top 20 for diversity
                item_counts[item] += 1
                total_recs += 1

        # Gini coefficient for recommendation fairness
        if total_recs == 0:
            gini = 0.0
        else:
            counts = sorted(item_counts.values())
            n = len(counts)
            if n == 0:
                gini = 0.0
            else:
                cumsum = np.cumsum(counts)
                gini = (n + 1 - 2 * sum((n + 1 - i) * counts[i] for i in range(n)) / cumsum[-1]) / n

        # Simple diversity metrics
        unique_items_per_user = [len(set(recs[:20])) for recs in recommendations]
        avg_unique_items = np.mean(unique_items_per_user) if unique_items_per_user else 0

        return {
            'avg_unique_items_per_user': avg_unique_items,
            'gini_coefficient': gini,
            'total_unique_items_recommended': len(item_counts)
        }

    def evaluate_model_performance(self, model_name: str, recommendations: Dict[int, List[int]],
                                 test_data: Dict[int, Set[int]], total_items: int) -> Dict:
        """Evaluate a model's performance with comprehensive metrics."""

        logging.info(f"Evaluating {model_name} performance...")

        metrics = {
            'model_name': model_name,
            'total_users_evaluated': len(recommendations),
            'total_items_in_catalog': total_items
        }

        # Initialize metric accumulators
        for k in self.k_values:
            metrics[f'recall@{k}'] = []
            metrics[f'precision@{k}'] = []
            metrics[f'ndcg@{k}'] = []

        mrr_scores = []
        all_recommendations = []

        # Evaluate each user
        for user_id, recommended_items in recommendations.items():
            relevant_items = test_data.get(user_id, set())

            if not relevant_items:
                continue  # Skip users with no test items

            all_recommendations.append(recommended_items)

            # Calculate MRR
            mrr_scores.append(self.mrr(recommended_items, relevant_items))

            # Calculate metrics for each K
            for k in self.k_values:
                recall_k = self.recall_at_k(recommended_items, relevant_items, k)
                precision_k = self.precision_at_k(recommended_items, relevant_items, k)
                ndcg_k = self.ndcg_at_k(recommended_items, relevant_items, k)

                metrics[f'recall@{k}'].append(recall_k)
                metrics[f'precision@{k}'].append(precision_k)
                metrics[f'ndcg@{k}'].append(ndcg_k)

        # Aggregate metrics
        for k in self.k_values:
            metrics[f'recall@{k}_mean'] = np.mean(metrics[f'recall@{k}'])
            metrics[f'recall@{k}_std'] = np.std(metrics[f'recall@{k}'])
            metrics[f'precision@{k}_mean'] = np.mean(metrics[f'precision@{k}'])
            metrics[f'precision@{k}_std'] = np.std(metrics[f'precision@{k}'])
            metrics[f'ndcg@{k}_mean'] = np.mean(metrics[f'ndcg@{k}'])
            metrics[f'ndcg@{k}_std'] = np.std(metrics[f'ndcg@{k}'])

            # Remove individual scores to keep output manageable
            del metrics[f'recall@{k}']
            del metrics[f'precision@{k}']
            del metrics[f'ndcg@{k}']

        # MRR
        metrics['mrr_mean'] = np.mean(mrr_scores) if mrr_scores else 0.0
        metrics['mrr_std'] = np.std(mrr_scores) if mrr_scores else 0.0

        # Coverage and diversity
        metrics['catalog_coverage'] = self.catalog_coverage(all_recommendations, total_items)
        diversity_metrics = self.diversity_metrics(all_recommendations)
        metrics.update(diversity_metrics)

        return metrics

class TwoTowerEvaluator:
    """Evaluator for Two-Tower model."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """Load the trained Two-Tower model."""
        logging.info("Loading Two-Tower model...")

        # Load data
        self.data_loader = TwoTowerDataLoader(
            sequences_path="data/processed/sequences_with_metadata.parquet",
            movies_path="data/processed/content_features.parquet",
            negative_sampling_ratio=1,
            min_user_interactions=1
        )

        # Load model config
        with open(self.model_dir / "model_config.json", 'r') as f:
            config = json.load(f)

        # Initialize model
        self.model = TwoTowerModel(
            num_users=config['num_users'],
            num_movies=config['num_movies'],
            content_feature_dim=config['content_feature_dim'],
            embedding_dim=config['training_config']['model']['embedding_dim'],
            user_hidden_dims=config['training_config']['model']['user_hidden_dims'],
            item_hidden_dims=config['training_config']['model']['item_hidden_dims'],
            dropout_rate=config['training_config']['model']['dropout_rate']
        )

        # Load weights
        self.model.load_state_dict(torch.load(
            self.model_dir / "two_tower_model.pth", map_location=self.device
        ))
        self.model.eval()

        logging.info("Two-Tower model loaded successfully")

    def generate_recommendations(self, user_ids: List[int], k: int = 100) -> Dict[int, List[int]]:
        """Generate recommendations for given users."""
        recommendations = {}

        with torch.no_grad():
            # Get all movie embeddings once
            movie_indices = torch.arange(self.model.num_movies)
            movie_embs = self.model.get_movie_embeddings(movie_indices, self.data_loader.movie_features)

            for user_id in tqdm(user_ids, desc="Generating Two-Tower recommendations"):
                if user_id not in self.data_loader.user_to_idx:
                    continue

                user_idx = self.data_loader.user_to_idx[user_id]
                user_emb = self.model.get_user_embeddings(torch.tensor([user_idx]))

                # Calculate similarities
                scores = torch.mm(user_emb, movie_embs.T).squeeze()
                top_indices = torch.argsort(scores, descending=True)[:k]

                # Convert to original movie IDs
                recommended_movies = [self.data_loader.idx_to_movie[idx.item()] for idx in top_indices]
                recommendations[user_id] = recommended_movies

        return recommendations

class BERT4RecEvaluator:
    """Evaluator for BERT4Rec model."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """Load the trained BERT4Rec model."""
        logging.info("Loading BERT4Rec model...")

        # Load data
        self.data_loader = BERT4RecDataLoader(
            sequences_path="data/processed/sequences_with_metadata.parquet",
            movies_path="data/processed/content_features.parquet",
            max_seq_len=50,
            min_seq_len=3
        )

        # Load model config
        with open(self.model_dir / "bert4rec_config.json", 'r') as f:
            config = json.load(f)

        # Initialize model
        self.model = BERT4RecModel(
            num_items=config['num_items'],
            content_feature_dim=config['content_feature_dim'],
            max_seq_len=config['max_seq_len'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout_rate=config['training_config']['model']['dropout_rate']
        )

        # Load weights
        self.model.load_state_dict(torch.load(
            self.model_dir / "bert4rec_model.pth", map_location=self.device
        ))
        self.model.eval()

        logging.info("BERT4Rec model loaded successfully")

    def generate_recommendations(self, user_ids: List[int], k: int = 100) -> Dict[int, List[int]]:
        """Generate recommendations for given users."""
        recommendations = {}

        with torch.no_grad():
            for user_id in tqdm(user_ids, desc="Generating BERT4Rec recommendations"):
                if user_id not in self.data_loader.user_to_idx:
                    continue

                user_idx = self.data_loader.user_to_idx[user_id]
                user_interactions = self.data_loader.user_sequences.get(user_idx, [])

                if len(user_interactions) == 0:
                    continue

                # Extract movie indices from interaction records
                movie_sequence = [interaction['movie_idx'] for interaction in user_interactions]

                # Get already seen movies for filtering
                seen_movie_ids = set()
                for interaction in user_interactions:
                    if 'movie_id' in interaction:
                        seen_movie_ids.add(interaction['movie_id'])
                    elif 'movieId' in interaction:
                        seen_movie_ids.add(interaction['movieId'])

                if len(movie_sequence) == 0:
                    continue

                # Prepare input sequence (last 49 items + mask token)
                input_sequence = movie_sequence[-49:] + [0]  # 0 is mask token

                # Pad to max_seq_len if needed
                while len(input_sequence) < self.model.max_seq_len:
                    input_sequence = [0] + input_sequence  # Pad at beginning

                # Truncate if too long
                input_sequence = input_sequence[-self.model.max_seq_len:]

                seq_tensor = torch.tensor([input_sequence], dtype=torch.long)

                # Create features for the sequence
                seq_features = torch.zeros(1, self.model.max_seq_len, self.data_loader.movie_features.shape[1])

                # Fill in actual movie features where available
                for i, movie_idx in enumerate(input_sequence):
                    if movie_idx > 0 and movie_idx in self.data_loader.movie_feature_map:
                        feature_idx = self.data_loader.movie_feature_map[movie_idx]
                        if feature_idx < len(self.data_loader.movie_features):
                            seq_features[0, i] = self.data_loader.movie_features[feature_idx]

                # Get predictions for the masked position (last position)
                logits = self.model(seq_tensor, seq_features)
                scores = torch.softmax(logits[0, -1], dim=-1)  # Last position (masked)

                # Get top-k items
                top_indices = torch.argsort(scores, descending=True)[:k*2]  # Get more to filter

                # Convert to original movie IDs and filter out padding/mask tokens
                recommended_movies = []
                for idx in top_indices:
                    idx_val = idx.item()
                    if idx_val > 0 and idx_val in self.data_loader.idx_to_movie:
                        movie_id = self.data_loader.idx_to_movie[idx_val]
                        if movie_id not in seen_movie_ids:  # Not already seen
                            recommended_movies.append(movie_id)
                        if len(recommended_movies) >= k:
                            break

                recommendations[user_id] = recommended_movies[:k]

        return recommendations

def create_test_split(data_loader, split_ratio: float = 0.2) -> Tuple[Dict[int, Set[int]], List[int]]:
    """Create a temporal test split for evaluation."""
    logging.info(f"Creating temporal test split ({split_ratio:.0%})...")

    # Load the sequences dataframe
    sequences_df = pd.read_parquet("data/processed/sequences_with_metadata.parquet")

    # Sort by datetime for temporal split
    if 'datetime' in sequences_df.columns:
        sequences_df = sequences_df.sort_values('datetime')

    # Split point
    split_idx = int(len(sequences_df) * (1 - split_ratio))
    test_sequences = sequences_df.iloc[split_idx:]

    # Create test data: {user_id: set(relevant_movie_ids)}
    test_data = defaultdict(set)
    test_user_ids = []

    for _, row in test_sequences.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row.get('thumbs_rating', row.get('rating', 0))

        # Only consider positive interactions as relevant
        if rating >= 1.0:  # thumbs_rating 1.0 (up) or 2.0 (double up)
            test_data[user_id].add(movie_id)
            if user_id not in test_user_ids:
                test_user_ids.append(user_id)

    logging.info(f"Created test set: {len(test_user_ids)} users, {sum(len(items) for items in test_data.values())} positive interactions")

    return dict(test_data), test_user_ids[:500]  # Limit to 500 users for faster evaluation

def main():
    """Main evaluation script."""
    logging.info("Starting comprehensive model evaluation...")

    # Create test split
    # We need a way to create test data - using a simple temporal split
    data_loader = TwoTowerDataLoader(
        sequences_path="data/processed/sequences_with_metadata.parquet",
        movies_path="data/processed/content_features.parquet",
        negative_sampling_ratio=1,
        min_user_interactions=1
    )

    test_data, test_user_ids = create_test_split(data_loader)

    # Initialize evaluators
    tt_evaluator = TwoTowerEvaluator("models/two_tower")
    bert_evaluator = BERT4RecEvaluator("models/bert4rec")

    # Load models
    tt_evaluator.load_model()
    bert_evaluator.load_model()

    # Generate recommendations
    logging.info("Generating recommendations...")
    tt_recommendations = tt_evaluator.generate_recommendations(test_user_ids, k=100)
    bert_recommendations = bert_evaluator.generate_recommendations(test_user_ids, k=100)

    # Evaluate models
    evaluator = RecommendationEvaluator(k_values=[5, 10, 20, 50])

    tt_metrics = evaluator.evaluate_model_performance(
        "Two-Tower", tt_recommendations, test_data, data_loader.num_movies
    )

    bert_metrics = evaluator.evaluate_model_performance(
        "BERT4Rec", bert_recommendations, test_data, data_loader.num_movies
    )

    # Combine results
    results = {
        'evaluation_info': {
            'test_users': len(test_user_ids),
            'test_interactions': sum(len(items) for items in test_data.values()),
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'k_values': evaluator.k_values
        },
        'two_tower_metrics': tt_metrics,
        'bert4rec_metrics': bert_metrics
    }

    # Save results
    output_path = Path("results/individual_model_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL EVALUATION RESULTS")
    print("="*80)

    models = [('Two-Tower', tt_metrics), ('BERT4Rec', bert_metrics)]

    for model_name, metrics in models:
        print(f"\n{model_name.upper()} PERFORMANCE:")
        print(f"  Recall@10:    {metrics['recall@10_mean']:.4f} (±{metrics['recall@10_std']:.4f})")
        print(f"  Precision@10: {metrics['precision@10_mean']:.4f} (±{metrics['precision@10_std']:.4f})")
        print(f"  nDCG@10:      {metrics['ndcg@10_mean']:.4f} (±{metrics['ndcg@10_std']:.4f})")
        print(f"  MRR:          {metrics['mrr_mean']:.4f} (±{metrics['mrr_std']:.4f})")
        print(f"  Coverage:     {metrics['catalog_coverage']:.4f}")
        print(f"  Avg Unique:   {metrics['avg_unique_items_per_user']:.1f} items/user")

    print(f"\n📄 Detailed results saved to: {output_path}")

    logging.info("Individual model evaluation completed!")

if __name__ == "__main__":
    main()