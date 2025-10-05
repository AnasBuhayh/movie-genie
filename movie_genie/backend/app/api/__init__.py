"""
API Blueprint for Movie Genie

This module organizes all API endpoints into a single blueprint.
"""

from flask import Blueprint, jsonify

# Create the main API blueprint
api_bp = Blueprint('api', __name__)

# Import all API modules (these will register their routes with sub-blueprints)
from . import search
from . import movies
from . import recommendations
from . import feedback
from . import users
from . import models_routes

# Register sub-blueprints
api_bp.register_blueprint(search.search_bp, url_prefix='/search')
api_bp.register_blueprint(movies.movies_bp, url_prefix='/movies')
api_bp.register_blueprint(recommendations.recommendations_bp, url_prefix='/recommendations')
api_bp.register_blueprint(feedback.feedback_bp, url_prefix='/feedback')
api_bp.register_blueprint(users.users_bp, url_prefix='/users')
api_bp.register_blueprint(models_routes.models_bp, url_prefix='/models')

@api_bp.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Movie Genie API is running',
        'version': '1.0.0'
    })

@api_bp.route('/')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Movie Genie API',
        'version': '1.0.0',
        'description': 'AI-powered movie recommendations with semantic search and personalized recommendations',
        'endpoints': {
            'search': '/api/search/',
            'movies': '/api/movies/',
            'recommendations': '/api/recommendations/',
            'feedback': '/api/feedback/',
            'users': '/api/users/'
        }
    })