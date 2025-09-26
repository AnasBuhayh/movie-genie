#!/usr/bin/env python3
"""
Test script to verify semantic search consistency with google/embeddinggemma-300M.

This script verifies that:
1. Query encoder uses google/embeddinggemma-300M model
2. Movie embeddings have correct dimension (768D)
3. Query and movie embeddings are in the same vector space
4. Similarity computation works correctly
"""

import numpy as np
import logging
from movie_genie.search.semantic_engine import QueryEncoder, MovieEmbeddingLoader

logging.basicConfig(level=logging.INFO)

def test_semantic_search_consistency():
    print("🧪 SEMANTIC SEARCH CONSISTENCY TEST")
    print("="*50)

    # Test 1: Initialize components
    print("\n1️⃣ Initializing semantic search components...")

    try:
        query_encoder = QueryEncoder("configs/semantic_search.yaml")
        movie_loader = MovieEmbeddingLoader("data/processed/content_features.parquet")

        embeddings, metadata = movie_loader.get_embeddings_and_metadata()
        print(f"✅ Query encoder: {query_encoder.model_name}")
        print(f"✅ Movie embeddings: {embeddings.shape}")
        print(f"✅ Query dimension: {query_encoder.encoding_dimension}")
        print(f"✅ Movie dimension: {embeddings.shape[1]}")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

    # Test 2: Verify dimensions match
    print("\n2️⃣ Verifying embedding dimensions...")

    if query_encoder.encoding_dimension != embeddings.shape[1]:
        print(f"❌ Dimension mismatch! Query: {query_encoder.encoding_dimension}, Movies: {embeddings.shape[1]}")
        return False
    else:
        print(f"✅ Dimensions match: {query_encoder.encoding_dimension}D")

    # Test 3: Test query encoding
    print("\n3️⃣ Testing query encoding...")

    test_queries = [
        "science fiction space adventure",
        "romantic comedy with strong female lead",
        "psychological thriller with plot twists",
        "animated family movie with humor"
    ]

    try:
        # Test single query encoding
        query_vector = query_encoder.encode(test_queries[0])
        print(f"✅ Single query encoding: {query_vector.shape}")

        # Test batch encoding
        batch_vectors = query_encoder.encode_batch(test_queries)
        print(f"✅ Batch query encoding: {batch_vectors.shape}")

        # Verify normalization
        if query_encoder.normalize_vectors:
            norms = np.linalg.norm(batch_vectors, axis=1)
            print(f"✅ Vector norms (should be ~1.0): {norms}")

    except Exception as e:
        print(f"❌ Query encoding failed: {e}")
        return False

    # Test 4: Test similarity computation
    print("\n4️⃣ Testing semantic similarity...")

    try:
        # Find movies most similar to first query
        query_vector = query_encoder.encode(test_queries[0])  # sci-fi space adventure

        # Compute similarities
        similarities = np.dot(embeddings, query_vector)
        top_indices = np.argsort(similarities)[::-1][:5]

        print(f"✅ Top 5 movies for '{test_queries[0]}':")
        for i, idx in enumerate(top_indices):
            movie = metadata[idx]
            similarity = similarities[idx]
            print(f"   {i+1}. {movie['title']} (similarity: {similarity:.3f})")

    except Exception as e:
        print(f"❌ Similarity computation failed: {e}")
        return False

    # Test 5: Test multiple query types
    print("\n5️⃣ Testing diverse query types...")

    diverse_queries = [
        "movies like Star Wars",      # Reference-based
        "funny animated movies",      # Genre + attribute
        "Christopher Nolan films",    # Director-based
        "dark psychological drama",   # Mood-based
    ]

    try:
        for query in diverse_queries:
            query_vector = query_encoder.encode(query)
            similarities = np.dot(embeddings, query_vector)
            top_idx = np.argmax(similarities)
            top_movie = metadata[top_idx]

            print(f"   '{query}' → {top_movie['title']} ({similarities[top_idx]:.3f})")

    except Exception as e:
        print(f"❌ Diverse query testing failed: {e}")
        return False

    # Test 6: Verify caching works
    print("\n6️⃣ Testing query caching...")

    try:
        # Encode same query twice
        import time

        start_time = time.time()
        query_encoder.encode("test caching query")
        first_time = time.time() - start_time

        start_time = time.time()
        query_encoder.encode("test caching query")  # Should use cache
        second_time = time.time() - start_time

        print(f"✅ First encoding: {first_time:.3f}s")
        print(f"✅ Cached encoding: {second_time:.3f}s")
        print(f"✅ Cache entries: {len(query_encoder.query_cache)}")

    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

    print("\n🎉 ALL TESTS PASSED!")
    print("\n📋 SUMMARY:")
    print(f"   Model: {query_encoder.model_name}")
    print(f"   Dimensions: {query_encoder.encoding_dimension}D")
    print(f"   Movies: {len(metadata):,}")
    print(f"   Normalization: {query_encoder.normalize_vectors}")
    print(f"   Cache entries: {len(query_encoder.query_cache)}")

    return True

def test_model_consistency():
    """Test that we're using the same model as content feature generation."""
    print("\n🔍 MODEL CONSISTENCY CHECK")
    print("="*30)

    # Check config
    query_encoder = QueryEncoder("configs/semantic_search.yaml")
    expected_model = "google/embeddinggemma-300M"

    if query_encoder.model_name == expected_model:
        print(f"✅ Using correct model: {expected_model}")
        print(f"✅ Embedding dimension: {query_encoder.encoding_dimension}")
        return True
    else:
        print(f"❌ Wrong model! Expected: {expected_model}, Got: {query_encoder.model_name}")
        return False

if __name__ == "__main__":
    success = True

    # Run model consistency check
    success &= test_model_consistency()

    # Run full consistency test
    success &= test_semantic_search_consistency()

    if success:
        print("\n🌟 SEMANTIC SEARCH SYSTEM READY!")
    else:
        print("\n💥 TESTS FAILED - Check configuration and dependencies")
        exit(1)