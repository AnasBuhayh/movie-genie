#!/usr/bin/env python3
"""
Movie Genie Flask Backend

Main entry point for the Movie Genie Flask application.
Integrates with your existing ML infrastructure and database layer.
"""

import os
import sys
import signal
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from movie_genie.backend.app import create_app

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print("\n\n👋 Movie Genie Backend shutting down gracefully...")
    sys.exit(0)

def main():
    """Main entry point"""

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination

    # Create Flask app
    app = create_app()

    # Configuration
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    print(f"""
🎬 Movie Genie Backend Starting...

🌐 Server: http://{host}:{port}
🔧 Environment: {os.environ.get('FLASK_ENV', 'development')}
🎯 API Base URL: http://{host}:{port}/api
📱 Frontend: http://{host}:{port}

📋 Available endpoints:
   • GET  /api/health              - Health check
   • GET  /api/search/semantic     - Semantic movie search
   • GET  /api/movies/<id>         - Movie details
   • GET  /api/movies/popular      - Popular movies
   • POST /api/recommendations/personalized - Personalized recommendations
   • POST /api/feedback            - Submit user feedback

🏗️  Frontend build status:
   • Templates: {'✅' if (Path(__file__).parent / 'templates' / 'index.html').exists() else '❌'}
   • Static files: {'✅' if (Path(__file__).parent / 'static').exists() else '❌'}

💡 To build frontend: cd movie_genie/frontend && npm run build

🚀 Starting Flask server... (Press Ctrl+C to stop)
    """)

    try:
        # Use use_reloader=False to prevent double signal handling in debug mode
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # Prevents duplicate processes and cleaner shutdown
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()