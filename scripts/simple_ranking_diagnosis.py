#!/usr/bin/env python3
"""
Simple diagnostic script to identify ranking correlation issues.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

# Add project root to path
import sys
sys.path.append('.')

from movie_genie.retrieval.two_tower_model import TwoTowerDataLoader
from movie_genie.ranking.bert4rec_model import BERT4RecDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_data_differences():
    """Analyze data differences between Two-Tower and BERT4Rec."""
    logging.info("Loading Two-Tower data...")

    tt_loader = TwoTowerDataLoader(
        sequences_path="data/processed/sequences_with_metadata.parquet",
        movies_path="data/processed/content_features.parquet",
        negative_sampling_ratio=1,
        min_user_interactions=1
    )

    logging.info("Loading BERT4Rec data...")

    bert_loader = BERT4RecDataLoader(
        sequences_path="data/processed/sequences_with_metadata.parquet",
        movies_path="data/processed/movies_with_content_features.parquet",
        max_seq_len=50,
        min_seq_len=3
    )

    # Compare data
    analysis = {}

    # User comparison
    tt_users = set(tt_loader.user_to_idx.keys())
    bert_users = set(bert_loader.user_to_idx.keys())

    analysis['users'] = {
        'tt_count': len(tt_users),
        'bert_count': len(bert_users),
        'common': len(tt_users & bert_users),
        'tt_only': len(tt_users - bert_users),
        'bert_only': len(bert_users - tt_users),
        'tt_only_users': list(tt_users - bert_users)[:10],  # Sample
        'bert_only_users': list(bert_users - tt_users)[:10]  # Sample
    }

    # Movie comparison
    tt_movies = set(tt_loader.movie_to_idx.keys())
    bert_movies = set(bert_loader.movie_to_idx.keys())

    analysis['movies'] = {
        'tt_count': len(tt_movies),
        'bert_count': len(bert_movies),
        'common': len(tt_movies & bert_movies),
        'tt_only': len(tt_movies - bert_movies),
        'bert_only': len(bert_movies - tt_movies),
        'tt_only_movies': list(tt_movies - bert_movies)[:20],  # Sample
        'bert_only_movies': list(bert_movies - tt_movies)[:20]  # Sample
    }

    # Data loading differences
    logging.info("Analyzing data loading parameters...")

    analysis['data_params'] = {
        'tt_min_interactions': 1,  # What we set for TT
        'bert_min_seq_len': 3,     # What BERT4Rec uses
        'tt_negative_sampling': 1,
        'sequences_path_same': True,
        'movies_path_different': tt_loader.movies_path != bert_loader.movies_path
    }

    # Check if the movie paths are actually different
    analysis['data_params']['tt_movies_path'] = "data/processed/content_features.parquet"
    analysis['data_params']['bert_movies_path'] = "data/processed/movies_with_content_features.parquet"
    analysis['data_params']['movies_path_different'] = True  # They are different!

    return analysis

def main():
    """Main analysis."""
    print("🔍 DIAGNOSING RANKING CORRELATION ISSUES")
    print("="*50)

    analysis = analyze_data_differences()

    print(f"\n📊 USER COMPARISON:")
    print(f"  Two-Tower users: {analysis['users']['tt_count']:,}")
    print(f"  BERT4Rec users:  {analysis['users']['bert_count']:,}")
    print(f"  Common users:    {analysis['users']['common']:,}")
    print(f"  TT-only users:   {analysis['users']['tt_only']:,}")
    print(f"  BERT-only users: {analysis['users']['bert_only']:,}")

    print(f"\n🎬 MOVIE COMPARISON:")
    print(f"  Two-Tower movies: {analysis['movies']['tt_count']:,}")
    print(f"  BERT4Rec movies:  {analysis['movies']['bert_count']:,}")
    print(f"  Common movies:    {analysis['movies']['common']:,}")
    print(f"  TT-only movies:   {analysis['movies']['tt_only']:,}")
    print(f"  BERT-only movies: {analysis['movies']['bert_only']:,}")

    print(f"\n🗂️ DATA SOURCE ANALYSIS:")
    print(f"  Two-Tower movies path:  {analysis['data_params']['tt_movies_path']}")
    print(f"  BERT4Rec movies path:   {analysis['data_params']['bert_movies_path']}")
    print(f"  Using different files:  {analysis['data_params']['movies_path_different']}")

    if analysis['movies']['tt_only'] > 0:
        print(f"\n⚠️  CRITICAL ISSUE FOUND:")
        print(f"   Two-Tower has {analysis['movies']['tt_only']} movies that BERT4Rec doesn't have!")
        print(f"   Sample TT-only movies: {analysis['movies']['tt_only_movies'][:5]}")

    if analysis['movies']['bert_only'] > 0:
        print(f"\n⚠️  CRITICAL ISSUE FOUND:")
        print(f"   BERT4Rec has {analysis['movies']['bert_only']} movies that Two-Tower doesn't have!")
        print(f"   Sample BERT-only movies: {analysis['movies']['bert_only_movies'][:5]}")

    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    if analysis['data_params']['movies_path_different']:
        print("   ❌ MODELS ARE USING DIFFERENT MOVIE DATA FILES!")
        print("   ❌ Two-Tower: content_features.parquet")
        print("   ❌ BERT4Rec:  movies_with_content_features.parquet")
        print("\n💡 SOLUTION:")
        print("   ✅ Make both models use the same movie data file")
        print("   ✅ Update dvc.yaml dependencies to be consistent")

    # Calculate impact
    overlap_pct = analysis['movies']['common'] / max(analysis['movies']['tt_count'], analysis['movies']['bert_count']) * 100
    print(f"\n📈 IMPACT ASSESSMENT:")
    print(f"   Movie vocabulary overlap: {overlap_pct:.1f}%")

    if overlap_pct < 95:
        print(f"   ❌ Low overlap explains negative correlation!")
        print(f"   ❌ Models ranking completely different item sets")

    # Save detailed analysis
    output_path = Path("results/simple_diagnosis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()