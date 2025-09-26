"""
Flask Application Factory for Movie Genie Backend

This creates the Flask app and integrates with your existing database layer.
"""

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import logging
from pathlib import Path

# Import your existing database setup
from .database.database import DatabaseManager
from ..config import get_config

def create_app(config_name=None):
    """
    Flask application factory

    Args:
        config_name: Configuration to use ('development', 'production', 'testing')

    Returns:
        Flask app instance
    """
    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static')

    # Load configuration
    config_class = get_config()
    app.config.from_object(config_class)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize CORS for frontend communication
    CORS(app, origins=app.config['CORS_ORIGINS'])

    # Initialize your existing database manager
    app.db_manager = DatabaseManager(app.config['SQLALCHEMY_DATABASE_URI'])

    # Create tables if they don't exist
    with app.app_context():
        app.db_manager.create_all_tables()

    # Register blueprints (API routes)
    register_blueprints(app)

    # Register frontend routes
    register_frontend_routes(app)

    # Register error handlers
    register_error_handlers(app)

    app.logger.info(f"Movie Genie Flask app created with {config_class.__name__}")
    return app

def register_blueprints(app):
    """Register API blueprints"""

    # Import and register API blueprints
    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    app.logger.info("API blueprints registered")

def register_frontend_routes(app):
    """Register routes for serving the React frontend"""

    @app.route('/')
    def index():
        """Serve the React app"""
        try:
            return render_template('index.html')
        except Exception as e:
            app.logger.error(f"Error serving frontend: {e}")
            return f"Frontend not built yet. Run 'npm run build' in the frontend directory.", 404

    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files (CSS, JS, images)"""
        return send_from_directory(app.static_folder, filename)

    # Handle React Router (SPA routing)
    @app.route('/<path:path>')
    def react_routes(path):
        """Handle React Router paths by serving index.html"""
        # Don't interfere with API routes
        if path.startswith('api/'):
            return "API route not found", 404

        try:
            return render_template('index.html')
        except Exception as e:
            app.logger.error(f"Error serving SPA route {path}: {e}")
            return f"Frontend not built yet. Run 'npm run build' in the frontend directory.", 404

def register_error_handlers(app):
    """Register error handlers for better API responses"""

    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500

    @app.errorhandler(400)
    def bad_request(error):
        return {'error': 'Bad request'}, 400