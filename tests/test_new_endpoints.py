#!/usr/bin/env python3
"""
Test script for the new Historical Interest and Watched Movies endpoints
"""

import requests
import json

API_BASE = "http://127.0.0.1:5001/api"
TEST_USER_ID = 1  # Use a valid user ID from your dataset

def test_endpoint(name, url, method="GET"):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"Method: {method}")

    try:
        if method == "GET":
            response = requests.get(url, timeout=10)

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success: ✅")

            # Pretty print the response structure
            print(f"\nResponse structure:")
            print(f"  - success: {result.get('success')}")
            print(f"  - message: {result.get('message')}")

            if 'data' in result:
                data = result['data']
                print(f"\nData structure:")
                for key, value in data.items():
                    if key == 'movies':
                        print(f"  - movies: {len(value)} movies")
                        if value:
                            print(f"    First movie: {value[0].get('title', 'No title')}")
                    elif isinstance(value, (list, dict)):
                        print(f"  - {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"  - {key}: {value}")
        else:
            print(f"Failed: ❌")
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"Error: ❌ {e}")

def main():
    """Run all endpoint tests"""
    print("🎬 Testing New Movie Genie Endpoints")
    print(f"API Base: {API_BASE}")
    print(f"Test User ID: {TEST_USER_ID}")

    # Test 1: User's Watched Movies
    test_endpoint(
        "User's Watched Movies",
        f"{API_BASE}/users/{TEST_USER_ID}/watched?limit=10"
    )

    # Test 2: User's Historical Interest
    test_endpoint(
        "User's Historical Interest",
        f"{API_BASE}/users/{TEST_USER_ID}/historical-interest?limit=10"
    )

    print(f"\n{'='*60}")
    print("✅ Testing complete!")
    print("\nNext steps:")
    print("1. Verify both endpoints return real data")
    print("2. Check that Historical Interest analyzes user's genre preferences")
    print("3. Confirm Watched Movies returns user's interaction history")
    print("4. Test with frontend: npm run dev in movie_genie/frontend")

if __name__ == "__main__":
    main()
