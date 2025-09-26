# Test script - save as scripts/test_embedding_loader.py
from movie_genie.search.semantic_engine import MovieEmbeddingLoader

def test_embedding_loading():
    loader = MovieEmbeddingLoader('data/processed/content_features.parquet')
    
    embeddings, metadata = loader.get_embeddings_and_metadata()
    stats = loader.get_statistics()
    
    print("Embedding loading test results:")
    print(f"Loaded {stats['total_movies']} movies")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")
    print("\nSample movies:")
    for movie in stats['sample_titles']:
        print(f"  - {movie}")

if __name__ == "__main__":
    test_embedding_loading()