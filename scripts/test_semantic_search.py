"""Test the complete semantic search engine."""

import sys
sys.path.append('.')
from movie_genie.search.semantic_engine import SemanticSearchEngine

def test_semantic_search():
    """Test basic semantic search functionality."""
    engine = SemanticSearchEngine()
    
    test_queries = [
        "science fiction movies about artificial intelligence",
        "romantic comedies with happy endings",
        "psychological thriller films",
        "movies like Blade Runner"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = engine.search(query, k=3)
        
        for result in results:
            print(f"  {result['rank']}. {result['title']} (Score: {result['similarity_score']:.3f})")
            print(f"     {result['overview'][:100]}...")
    
    return engine

def test_movie_similarity():
    """Test finding similar movies."""
    engine = SemanticSearchEngine()
    
    similar = engine.get_similar_movies("Blade Runner", k=5)
    print(f"\n🎬 Movies similar to 'Blade Runner':")
    
    for movie in similar:
        print(f"  {movie['rank']}. {movie['title']} (Score: {movie['similarity_score']:.3f})")

if __name__ == "__main__":
    engine = test_semantic_search()
    test_movie_similarity()
    
    stats = engine.get_engine_stats()
    print(f"\n📊 Engine stats: {stats['total_movies']} movies, {stats['embedding_dimension']}D vectors")