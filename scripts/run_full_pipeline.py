#!/usr/bin/env python3
"""
Run Complete Movie Genie Pipeline

This script runs the complete Movie Genie pipeline:
1. Data ingestion and processing
2. Model training (Two-Tower + BERT4Rec)
3. System evaluation
4. Web application deployment

Usage:
    python scripts/run_full_pipeline.py [--stage STAGE] [--web-only]

Examples:
    # Run complete pipeline
    python scripts/run_full_pipeline.py

    # Run only training stages
    python scripts/run_full_pipeline.py --stage training

    # Skip training, just run web app
    python scripts/run_full_pipeline.py --web-only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_dvc_stage(stage_name, force=False):
    """Run a specific DVC stage"""
    print(f"\n🔧 Running DVC stage: {stage_name}")

    cmd = ["dvc", "repro", stage_name]
    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Stage '{stage_name}' completed successfully")
        if result.stdout:
            print("📋 Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Stage '{stage_name}' failed: {e}")
        if e.stderr:
            print("💥 Error output:")
            print(e.stderr)
        return False

def run_pipeline_stage(stage):
    """Run specific pipeline stages"""

    if stage == "data":
        print("\n📊 Running data pipeline...")
        stages = ["ingest", "sequential_processing", "content_features"]

    elif stage == "training":
        print("\n🤖 Running model training...")
        stages = ["two_tower_training", "bert4rec_training"]

    elif stage == "evaluation":
        print("\n📈 Running system evaluation...")
        stages = ["integrated_evaluation"]

    elif stage == "frontend":
        print("\n🎨 Building frontend...")
        stages = ["frontend_build"]

    elif stage == "backend":
        print("\n🚀 Starting backend...")
        stages = ["backend_server"]

    elif stage == "web":
        print("\n🌐 Deploying web application...")
        stages = ["frontend_build", "backend_server"]

    else:
        print(f"❌ Unknown stage: {stage}")
        return False

    # Run each stage
    for stage_name in stages:
        if not run_dvc_stage(stage_name):
            return False

    return True

def run_complete_pipeline():
    """Run the complete pipeline from start to finish"""
    print("\n🎬 Running complete Movie Genie pipeline...")

    pipeline_stages = [
        ("data", "Data ingestion and processing"),
        ("training", "Model training (Two-Tower + BERT4Rec)"),
        ("evaluation", "System evaluation and metrics"),
        ("web", "Web application deployment")
    ]

    for stage, description in pipeline_stages:
        print(f"\n{'='*60}")
        print(f"🔄 Stage: {description}")
        print(f"{'='*60}")

        if not run_pipeline_stage(stage):
            print(f"\n❌ Pipeline failed at stage: {stage}")
            return False

        time.sleep(1)  # Brief pause between stages

    print(f"\n{'='*60}")
    print("🎉 Complete pipeline finished successfully!")
    print(f"{'='*60}")
    print("""
🌐 Your Movie Genie application is now running!

📱 Frontend: http://127.0.0.1:5000
🎯 API: http://127.0.0.1:5000/api
📊 Models: Two-Tower + BERT4Rec trained and ready
    """)

    return True

def show_pipeline_status():
    """Show current pipeline status"""
    print("\n📋 Movie Genie Pipeline Status")
    print("=" * 50)

    try:
        result = subprocess.run(["dvc", "status"], capture_output=True, text=True, check=False)
        if result.stdout.strip():
            print("🔄 Pipeline changes detected:")
            print(result.stdout)
        else:
            print("✅ All pipeline stages up to date")

        # Check if models exist
        models_dir = Path("models")
        if models_dir.exists():
            two_tower = models_dir / "two_tower"
            bert4rec = models_dir / "bert4rec"

            print("\n🤖 Model Status:")
            print(f"   Two-Tower: {'✅' if two_tower.exists() else '❌'}")
            print(f"   BERT4Rec:  {'✅' if bert4rec.exists() else '❌'}")

        # Check if web components exist
        backend_dir = Path("movie_genie/backend")
        frontend_templates = backend_dir / "templates" / "index.html"
        frontend_static = backend_dir / "static"

        print("\n🌐 Web Application Status:")
        print(f"   Backend:   {'✅' if backend_dir.exists() else '❌'}")
        print(f"   Frontend:  {'✅' if frontend_templates.exists() else '❌'}")
        print(f"   Static:    {'✅' if frontend_static.exists() else '❌'}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Could not check pipeline status: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Movie Genie Pipeline")
    parser.add_argument("--stage",
                       choices=["data", "training", "evaluation", "frontend", "backend", "web"],
                       help="Run specific pipeline stage")
    parser.add_argument("--web-only", action="store_true",
                       help="Skip training, just run web application")
    parser.add_argument("--status", action="store_true",
                       help="Show current pipeline status")
    parser.add_argument("--force", action="store_true",
                       help="Force re-run stages even if up to date")

    args = parser.parse_args()

    print("🎬 Movie Genie Pipeline Runner")
    print("=" * 50)

    # Show status if requested
    if args.status:
        show_pipeline_status()
        return

    # Run web-only mode
    if args.web_only:
        print("🌐 Running web application only (skipping training)...")
        if run_pipeline_stage("web"):
            print("\n🎉 Web application started successfully!")
        else:
            sys.exit(1)
        return

    # Run specific stage
    if args.stage:
        if run_pipeline_stage(args.stage):
            print(f"\n✅ Stage '{args.stage}' completed successfully!")
        else:
            sys.exit(1)
        return

    # Run complete pipeline
    if not run_complete_pipeline():
        sys.exit(1)

if __name__ == "__main__":
    main()