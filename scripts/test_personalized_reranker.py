"""Test personalized search reranking with real user data."""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from movie_genie.search.semantic_engine import SemanticSearchEngine

def load_real_users(sequences_path: str = "data/processed/sequences_with_metadata.parquet") -> pd.DataFrame:
    """Load real user interaction data."""
    sequences_df = pd.read_parquet(sequences_path)
    return sequences_df

def get_sample_users(sequences_df: pd.DataFrame, n_users: int = 5) -> list:
    """Get sample users with sufficient interaction history."""
    user_counts = sequences_df['userId'].value_counts()
    # Select users with at least 10 interactions for meaningful testing
    active_users = user_counts[user_counts >= 10].index.tolist()
    return active_users[:n_users]

def format_user_context(sequences_df: pd.DataFrame, user_id: int) -> dict:
    """Format real user data for reranking context."""
    user_interactions = sequences_df[sequences_df['userId'] == user_id].sort_values('datetime')

    interaction_history = []
    for _, row in user_interactions.iterrows():
        interaction_history.append({
            'movie_idx': int(row['movieId']),  # Use actual movie ID
            'rating': float(row['thumbs_rating']),
            'datetime': row['datetime']
        })
    
    return {
        'user_id': user_id,
        'interaction_history': interaction_history
    }

def test_real_user_reranking():
    """Test reranking with real user data."""
    # Load real data
    sequences_df = load_real_users()
    sample_users = get_sample_users(sequences_df, n_users=3)
    
    engine = SemanticSearchEngine()
    query = "science fiction movies about artificial intelligence"
    
    print(f"Testing with real users: {sample_users}")
    
    for user_id in sample_users:
        user_context = format_user_context(sequences_df, user_id)
        user_history_count = len(user_context['interaction_history'])
        
        print(f"\nUser {user_id} ({user_history_count} interactions):")
        
        # Get anonymous results
        anonymous_results = engine.search(query, k=5)
        
        # Get personalized results
        personalized_results = engine.search(query, k=5, user_context=user_context)
        
        print("  Anonymous vs Personalized:")
        for i in range(5):
            anon_title = anonymous_results[i]['title'][:25].ljust(25)
            pers_title = personalized_results[i]['title'][:25].ljust(25)
            pers_score = personalized_results[i].get('personalized_score', 0)
            print(f"  {i+1}. {anon_title} | {pers_title} ({pers_score:.3f})")

def test_user_preference_impact():
    """Test how different user preferences affect reranking."""
    sequences_df = load_real_users()
    sample_users = get_sample_users(sequences_df, n_users=2)
    
    engine = SemanticSearchEngine()
    query = "action movies"
    
    print(f"\nComparing user preference impact:")
    
    for user_id in sample_users:
        user_context = format_user_context(sequences_df, user_id)
        
        # Analyze user's rating patterns
        ratings = [int['rating'] for int in user_context['interaction_history']]
        avg_rating = np.mean(ratings)
        
        results = engine.search(query, k=3, user_context=user_context)
        
        print(f"\nUser {user_id} (avg rating: {avg_rating:.2f}):")
        for result in results:
            score = result.get('personalized_score', result['similarity_score'])
            print(f"  {result['rank']}. {result['title']} ({score:.3f})")

if __name__ == "__main__":
    print("Testing with Real User Data")
    print("=" * 40)
    
    test_real_user_reranking()
    test_user_preference_impact()