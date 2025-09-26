# scripts/test_query_encoder.py
"""Test the query encoder with clean, production-style interface."""

import sys
import numpy as np
import yaml
sys.path.append('.')
from movie_genie.search.semantic_engine import QueryEncoder

def test_basic_usage():
    """Demonstrate the clean interface for basic query encoding."""
    # Simple initialization - configuration details are hidden
    encoder = QueryEncoder()
    
    # Clean encoding interface
    query = "science fiction movies about artificial intelligence"
    vector = encoder.encode(query)
    
    print(f"Query: {query}")
    print(f"Encoded to {vector.shape} vector")
    print(f"Vector magnitude: {np.linalg.norm(vector):.4f}")
    
    return encoder

def test_batch_encoding():
    """Show how batch encoding works with the clean interface."""
    encoder = QueryEncoder()
    
    queries = [
        "romantic comedies",
        "psychological thrillers", 
        "movies like Blade Runner",
        "Christopher Nolan films"
    ]
    
    # Clean batch encoding interface
    vectors = encoder.encode_batch(queries)
    
    print(f"\nEncoded {len(queries)} queries")
    print(f"Result shape: {vectors.shape}")
    
    for i, query in enumerate(queries):
        print(f"'{query}' -> {vectors[i].shape} vector")

if __name__ == "__main__":
    encoder = test_basic_usage()
    test_batch_encoding()
    
    # Show that configuration details are available when needed
    info = encoder.get_info()
    print(f"\nEncoder info: {info['model_name']}, {info['encoding_dimension']}D")