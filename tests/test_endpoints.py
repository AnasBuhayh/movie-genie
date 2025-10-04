#!/usr/bin/env python3
"""
Quick test script to verify backend API endpoints are working
"""

import requests
import json

API_BASE = "http://127.0.0.1:5001/api"

def test_endpoint(name, url, method="GET", data=None):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"Method: {method}")

    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success: ✅")
            print(f"Response keys: {list(result.keys())}")
            if 'data' in result:
                data_keys = list(result['data'].keys()) if isinstance(result['data'], dict) else type(result['data'])
                print(f"Data structure: {data_keys}")
        else:
            print(f"Failed: ❌")
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"Error: ❌ {e}")

def main():
    """Run all endpoint tests"""
    print("🎬 Movie Genie API Endpoint Tests")
    print(f"Testing API at: {API_BASE}")

    # Test 1: Popular Movies
    test_endpoint(
        "Popular Movies",
        f"{API_BASE}/movies/popular?limit=5"
    )

    # Test 2: Semantic Search
    test_endpoint(
        "Semantic Search",
        f"{API_BASE}/search/semantic?q=action movies with robots"
    )

    # Test 3: Movie Details
    test_endpoint(
        "Movie Details",
        f"{API_BASE}/movies/1"
    )

    # Test 4: Personalized Recommendations
    test_endpoint(
        "Personalized Recommendations",
        f"{API_BASE}/recommendations/personalized",
        method="POST",
        data={
            "user_id": "123",
            "interaction_history": [],
            "limit": 5
        }
    )

    # Test 5: Submit Feedback
    test_endpoint(
        "Submit Feedback",
        f"{API_BASE}/feedback",
        method="POST",
        data={
            "movie_id": 1,
            "rating": 4.5,
            "feedback_type": "rating"
        }
    )

    print(f"\n{'='*60}")
    print("✅ Testing complete!")
    print("\nNext steps:")
    print("1. Start backend: cd movie_genie/backend && python app.py")
    print("2. Start frontend: cd movie_genie/frontend && npm run dev")
    print("3. Visit: http://localhost:8080")

if __name__ == "__main__":
    main()
