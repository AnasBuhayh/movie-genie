#!/usr/bin/env python3
"""
Run Movie Genie Web Application

This script runs the complete Movie Genie web application including:
1. Frontend build (if needed)
2. Flask backend with all ML models

Usage:
    python scripts/run_web_app.py [--build-frontend] [--port 5000]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and handle errors"""
    print(f"🔧 Running: {cmd}")
    if cwd:
        print(f"   Working directory: {cwd}")

    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return None

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")

    # Check Python dependencies
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import torch
        print("✅ Python dependencies OK")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("💡 Run: pip install -r movie_genie/backend/requirements.txt")
        return False

    # Check if Node.js is available for frontend build
    result = run_command("npm --version", check=False)
    if result and result.returncode == 0:
        print("✅ Node.js/npm OK")
    else:
        print("⚠️  Node.js/npm not available (frontend build will be skipped)")

    return True

def build_frontend():
    """Build the React frontend"""
    print("\n🎨 Building frontend...")

    frontend_dir = Path("movie_genie/frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False

    # Install dependencies if needed
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        result = run_command("npm install", cwd=frontend_dir)
        if not result:
            return False

    # Build frontend
    print("🏗️ Building React app...")
    result = run_command("npm run build", cwd=frontend_dir)
    if not result:
        return False

    print("✅ Frontend build complete")
    return True

def start_backend(port=5000):
    """Start the Flask backend"""
    print(f"\n🚀 Starting Movie Genie backend on port {port}...")

    backend_dir = Path("movie_genie/backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False

    # Set environment variables
    env = os.environ.copy()
    env["FLASK_PORT"] = str(port)
    env["FLASK_ENV"] = "development"

    # Start Flask server
    try:
        print(f"""
🎬 Movie Genie Web Application Starting...

🌐 Server: http://127.0.0.1:{port}
🎯 API: http://127.0.0.1:{port}/api
📱 Frontend: http://127.0.0.1:{port}

Press Ctrl+C to stop the server
        """)

        subprocess.run(
            ["python", "app.py"],
            cwd=backend_dir,
            env=env,
            check=True
        )

    except KeyboardInterrupt:
        print("\n👋 Movie Genie backend stopping...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed to start: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Movie Genie Web Application")
    parser.add_argument("--build-frontend", action="store_true",
                       help="Build frontend before starting backend")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run the server on (default: 5000)")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip dependency checks")

    args = parser.parse_args()

    print("🎬 Movie Genie Web Application")
    print("=" * 50)

    # Check dependencies
    if not args.skip_checks and not check_dependencies():
        sys.exit(1)

    # Build frontend if requested
    if args.build_frontend:
        if not build_frontend():
            print("❌ Frontend build failed")
            sys.exit(1)

    # Start backend
    if not start_backend(args.port):
        sys.exit(1)

if __name__ == "__main__":
    main()