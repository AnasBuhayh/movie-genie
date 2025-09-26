#!/usr/bin/env python3
"""
Diagnostic script to investigate ranking correlation issues between Two-Tower and BERT4Rec models.

This script performs detailed analysis to identify why the models disagree on item rankings.
"""

import torch
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# Add project root to path
import sys
sys.path.append('.')

from movie_genie.retrieval.two_tower_model import TwoTowerDataLoader, TwoTowerModel
from movie_genie.ranking.bert4rec_model import BERT4RecDataLoader, BERT4RecModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RankingDiagnostic:
    """Diagnostic tool for analyzing ranking correlation issues."""

    def __init__(self, two_tower_model_dir: str, bert4rec_model_dir: str):
        self.two_tower_model_dir = Path(two_tower_model_dir)
        self.bert4rec_model_dir = Path(bert4rec_model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_models(self):
        """Load both trained models."""
        logging.info("Loading Two-Tower model...")

        # Load Two-Tower model
        self.tt_data_loader = TwoTowerDataLoader(
            sequences_path="data/processed/sequences_with_metadata.parquet",
            movies_path="data/processed/content_features.parquet",
            negative_sampling_ratio=1,
            min_user_interactions=1
        )

        # Load model config
        with open(self.two_tower_model_dir / "model_config.json", 'r') as f:
            tt_config = json.load(f)

        self.tt_model = TwoTowerModel(
            num_users=tt_config['num_users'],
            num_movies=tt_config['num_movies'],
            content_feature_dim=tt_config['content_feature_dim'],
            embedding_dim=tt_config['training_config']['model']['embedding_dim'],
            user_hidden_dims=tt_config['training_config']['model']['user_hidden_dims'],
            item_hidden_dims=tt_config['training_config']['model']['item_hidden_dims'],
            dropout_rate=tt_config['training_config']['model']['dropout_rate']
        )

        self.tt_model.load_state_dict(torch.load(
            self.two_tower_model_dir / "two_tower_model.pth", map_location=self.device
        ))
        self.tt_model.eval()

        logging.info("Loading BERT4Rec model...")

        # Load BERT4Rec model
        self.bert_data_loader = BERT4RecDataLoader(
            sequences_path="data/processed/sequences_with_metadata.parquet",
            movies_path="data/processed/movies_with_content_features.parquet",
            max_seq_len=50,
            min_seq_len=3
        )

        with open(self.bert4rec_model_dir / "bert4rec_config.json", 'r') as f:
            bert_config = json.load(f)

        self.bert_model = BERT4RecModel(
            num_items=bert_config['num_items'],
            content_feature_dim=bert_config['content_feature_dim'],
            max_seq_len=bert_config['max_seq_len'],
            hidden_dim=bert_config['hidden_dim'],
            num_layers=bert_config['num_layers'],
            num_heads=bert_config['num_heads'],
            dropout_rate=bert_config['training_config']['model']['dropout_rate']
        )

        self.bert_model.load_state_dict(torch.load(
            self.bert4rec_model_dir / "bert4rec_model.pth", map_location=self.device
        ))
        self.bert_model.eval()

        logging.info("Models loaded successfully")

    def analyze_data_overlap(self) -> Dict:
        """Analyze how much data overlap exists between the two models."""
        logging.info("Analyzing data overlap between models...")

        # Get user/item mappings
        tt_users = set(self.tt_data_loader.user_to_idx.keys())
        tt_movies = set(self.tt_data_loader.movie_to_idx.keys())

        bert_users = set(self.bert_data_loader.user_to_idx.keys())
        bert_movies = set(self.bert_data_loader.movie_to_idx.keys())

        user_overlap = len(tt_users & bert_users)
        movie_overlap = len(tt_movies & bert_movies)

        overlap_analysis = {
            'user_overlap': {
                'total_tt': len(tt_users),
                'total_bert': len(bert_users),
                'overlap': user_overlap,
                'overlap_pct': user_overlap / min(len(tt_users), len(bert_users)) * 100
            },
            'movie_overlap': {
                'total_tt': len(tt_movies),
                'total_bert': len(bert_movies),
                'overlap': movie_overlap,
                'overlap_pct': movie_overlap / min(len(tt_movies), len(bert_movies)) * 100
            }
        }

        logging.info(f"User overlap: {user_overlap}/{min(len(tt_users), len(bert_users))} ({overlap_analysis['user_overlap']['overlap_pct']:.1f}%)")
        logging.info(f"Movie overlap: {movie_overlap}/{min(len(tt_movies), len(bert_movies))} ({overlap_analysis['movie_overlap']['overlap_pct']:.1f}%)")

        return overlap_analysis

    def compare_sample_rankings(self, user_id: int, candidate_size: int = 50) -> Dict:
        """Compare rankings for a specific user with detailed analysis."""
        logging.info(f"Analyzing rankings for user {user_id} with {candidate_size} candidates...")

        # Get user index for both models
        if user_id not in self.tt_data_loader.user_to_idx:
            raise ValueError(f"User {user_id} not found in Two-Tower data")
        if user_id not in self.bert_data_loader.user_to_idx:
            raise ValueError(f"User {user_id} not found in BERT4Rec data")

        tt_user_idx = self.tt_data_loader.user_to_idx[user_id]
        bert_user_idx = self.bert_data_loader.user_to_idx[user_id]

        # Get user's history for context
        user_history = self.bert_data_loader.user_sequences.get(user_id, [])

        # Generate Two-Tower rankings for all movies
        with torch.no_grad():
            user_emb = self.tt_model.get_user_embeddings(torch.tensor([tt_user_idx]))

            # Get all movie embeddings
            movie_indices = torch.arange(self.tt_model.num_movies)
            movie_features = self.tt_data_loader.movie_features
            movie_embs = self.tt_model.get_movie_embeddings(movie_indices, movie_features)

            # Calculate similarities
            tt_scores = torch.mm(user_emb, movie_embs.T).squeeze()
            tt_top_indices = torch.argsort(tt_scores, descending=True)[:candidate_size]
            tt_top_scores = tt_scores[tt_top_indices]

        # Convert to original movie IDs
        tt_top_movies = [self.tt_data_loader.idx_to_movie[idx.item()] for idx in tt_top_indices]

        # Get BERT4Rec rankings for the same candidates
        candidate_bert_indices = []
        for movie_id in tt_top_movies:
            if movie_id in self.bert_data_loader.movie_to_idx:
                candidate_bert_indices.append(self.bert_data_loader.movie_to_idx[movie_id])

        if len(candidate_bert_indices) < len(tt_top_movies):
            logging.warning(f"Only {len(candidate_bert_indices)}/{len(tt_top_movies)} candidates found in BERT4Rec data")

        # Generate BERT4Rec scores
        with torch.no_grad():
            # Create input sequence for this user
            if user_history:
                seq_tensor = torch.tensor([user_history[-49:] + [0]])  # Last 49 items + mask
                seq_features = torch.zeros(1, 50, self.bert_model.content_feature_dim)

                # Get scores for candidates
                logits = self.bert_model(seq_tensor, seq_features, candidate_items=torch.tensor([candidate_bert_indices]))
                bert_scores = torch.softmax(logits, dim=-1)[0, -1]  # Last position scores
            else:
                # No history - use random scores
                bert_scores = torch.randn(len(candidate_bert_indices))

        # Compare rankings
        tt_ranks = list(range(len(tt_top_movies)))
        bert_ranks = torch.argsort(bert_scores, descending=True).tolist()

        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(tt_ranks[:len(bert_ranks)], bert_ranks)
        pearson_corr, pearson_p = pearsonr(tt_ranks[:len(bert_ranks)], bert_ranks)

        analysis = {
            'user_id': user_id,
            'user_history_length': len(user_history),
            'candidate_size': candidate_size,
            'tt_top_movies': tt_top_movies[:10],  # Top 10 for display
            'tt_top_scores': tt_top_scores[:10].tolist(),
            'bert_reranked_movies': [tt_top_movies[i] for i in bert_ranks[:10]],
            'bert_scores': bert_scores[:10].tolist(),
            'spearman_correlation': spearman_corr,
            'pearson_correlation': pearson_corr,
            'rank_differences': [(i, bert_ranks[i]) for i in range(min(10, len(bert_ranks)))]
        }

        return analysis

    def analyze_feature_importance(self) -> Dict:
        """Analyze what features each model emphasizes."""
        logging.info("Analyzing feature importance...")

        # Sample some users and items
        sample_users = list(self.tt_data_loader.user_to_idx.keys())[:50]
        sample_movies = list(self.tt_data_loader.movie_to_idx.keys())[:100]

        # Get embeddings from both models
        with torch.no_grad():
            # Two-Tower embeddings
            tt_user_indices = [self.tt_data_loader.user_to_idx[uid] for uid in sample_users if uid in self.tt_data_loader.user_to_idx]
            tt_movie_indices = [self.tt_data_loader.movie_to_idx[mid] for mid in sample_movies if mid in self.tt_data_loader.movie_to_idx]

            tt_user_embs = self.tt_model.get_user_embeddings(torch.tensor(tt_user_indices))
            tt_movie_embs = self.tt_model.get_movie_embeddings(
                torch.tensor(tt_movie_indices),
                self.tt_data_loader.movie_features[tt_movie_indices]
            )

            # Calculate embedding statistics
            tt_user_var = torch.var(tt_user_embs, dim=0).mean().item()
            tt_movie_var = torch.var(tt_movie_embs, dim=0).mean().item()

        analysis = {
            'two_tower': {
                'user_embedding_variance': tt_user_var,
                'movie_embedding_variance': tt_movie_var,
                'embedding_dim': self.tt_model.embedding_dim,
                'uses_content_features': True
            },
            'bert4rec': {
                'embedding_dim': self.bert_model.hidden_dim,
                'sequence_based': True,
                'uses_content_features': True,
                'max_sequence_length': self.bert_model.max_seq_len
            }
        }

        return analysis

    def run_full_diagnosis(self, output_path: str = "results/ranking_diagnosis.json"):
        """Run complete diagnostic analysis."""
        logging.info("Starting full ranking correlation diagnosis...")

        self.load_models()

        diagnosis = {
            'data_overlap': self.analyze_data_overlap(),
            'feature_analysis': self.analyze_feature_importance(),
            'sample_rankings': []
        }

        # Analyze rankings for several sample users
        sample_user_ids = list(self.tt_data_loader.user_to_idx.keys())[:10]

        for user_id in sample_user_ids:
            try:
                user_analysis = self.compare_sample_rankings(user_id, candidate_size=50)
                diagnosis['sample_rankings'].append(user_analysis)
                logging.info(f"User {user_id}: Spearman correlation = {user_analysis['spearman_correlation']:.3f}")
            except Exception as e:
                logging.warning(f"Failed to analyze user {user_id}: {e}")

        # Calculate overall statistics
        correlations = [r['spearman_correlation'] for r in diagnosis['sample_rankings'] if not np.isnan(r['spearman_correlation'])]

        diagnosis['summary'] = {
            'avg_spearman_correlation': np.mean(correlations) if correlations else 0,
            'std_spearman_correlation': np.std(correlations) if correlations else 0,
            'users_analyzed': len(diagnosis['sample_rankings']),
            'main_findings': self._generate_findings(diagnosis)
        }

        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(diagnosis, f, indent=2, default=str)

        logging.info(f"Diagnosis complete. Results saved to {output_path}")
        return diagnosis

    def _generate_findings(self, diagnosis: Dict) -> List[str]:
        """Generate key findings from the diagnosis."""
        findings = []

        # Data overlap findings
        user_overlap_pct = diagnosis['data_overlap']['user_overlap']['overlap_pct']
        movie_overlap_pct = diagnosis['data_overlap']['movie_overlap']['overlap_pct']

        if user_overlap_pct < 95:
            findings.append(f"User overlap is only {user_overlap_pct:.1f}% - models may be training on different user sets")

        if movie_overlap_pct < 95:
            findings.append(f"Movie overlap is only {movie_overlap_pct:.1f}% - models may have different item vocabularies")

        # Correlation findings
        correlations = [r['spearman_correlation'] for r in diagnosis['sample_rankings'] if not np.isnan(r['spearman_correlation'])]
        avg_corr = np.mean(correlations) if correlations else 0

        if avg_corr < 0:
            findings.append(f"Average correlation is negative ({avg_corr:.3f}) - models are ranking items in opposite orders")
        elif avg_corr < 0.3:
            findings.append(f"Low positive correlation ({avg_corr:.3f}) - models have different ranking preferences")

        # History length analysis
        history_lengths = [r['user_history_length'] for r in diagnosis['sample_rankings']]
        avg_history = np.mean(history_lengths) if history_lengths else 0

        if avg_history < 10:
            findings.append(f"Users have short histories (avg {avg_history:.1f}) - BERT4Rec may struggle with sparse data")

        return findings

def main():
    """Main diagnostic script."""
    diagnostic = RankingDiagnostic(
        two_tower_model_dir="models/two_tower",
        bert4rec_model_dir="models/bert4rec"
    )

    diagnosis = diagnostic.run_full_diagnosis()

    print("\n" + "="*60)
    print("RANKING CORRELATION DIAGNOSIS SUMMARY")
    print("="*60)

    print(f"\nData Overlap:")
    print(f"  Users: {diagnosis['data_overlap']['user_overlap']['overlap_pct']:.1f}%")
    print(f"  Movies: {diagnosis['data_overlap']['movie_overlap']['overlap_pct']:.1f}%")

    print(f"\nRanking Correlation:")
    print(f"  Average Spearman: {diagnosis['summary']['avg_spearman_correlation']:.3f}")
    print(f"  Std Dev: {diagnosis['summary']['std_spearman_correlation']:.3f}")

    print(f"\nKey Findings:")
    for finding in diagnosis['summary']['main_findings']:
        print(f"  • {finding}")

    print(f"\nRecommendations:")
    if diagnosis['summary']['avg_spearman_correlation'] < 0:
        print("  • Models are anti-correlated - check if they're optimizing for opposite objectives")
        print("  • Verify training data consistency between models")
        print("  • Consider joint training or knowledge distillation")

    print(f"\nDetailed results saved to: results/ranking_diagnosis.json")

if __name__ == "__main__":
    main()