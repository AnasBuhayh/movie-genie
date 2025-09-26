"""
Flask Configuration for Movie Genie Backend
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration with common settings"""

    # Basic Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Database config (using your existing setup)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///movie_genie.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # API Configuration
    API_TITLE = "Movie Genie API"
    API_VERSION = "v1"

    # ML Model Paths (relative to your project root)
    BASE_DIR = Path(__file__).parent.parent.parent  # Points to movie-genie/ root
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"

    # Model-specific paths
    BERT4REC_MODEL_PATH = MODELS_DIR / "bert4rec" / "bert4rec_model.pth"
    BERT4REC_ARTIFACTS_PATH = MODELS_DIR / "bert4rec" / "data_artifacts.pkl"
    TWO_TOWER_MODEL_PATH = MODELS_DIR / "two_tower" / "two_tower_model.pth"

    # Data paths
    CONTENT_FEATURES_PATH = DATA_DIR / "processed" / "content_features.parquet"
    SEQUENCES_PATH = DATA_DIR / "processed" / "sequences_with_metadata.parquet"

    # Semantic search config
    SEMANTIC_SEARCH_CONFIG = BASE_DIR / "configs" / "semantic_search.yaml"

    # Frontend serving
    STATIC_FOLDER = 'static'
    TEMPLATE_FOLDER = 'templates'

    # CORS settings
    CORS_ORIGINS = ["http://localhost:8080", "http://localhost:3000"]  # Frontend dev servers

    # Caching (optional)
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes

    # API Response settings
    JSONIFY_PRETTYPRINT_REGULAR = True
    JSON_SORT_KEYS = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

    # More verbose logging in development
    LOG_LEVEL = "DEBUG"

    # Allow all origins in development for easier testing
    CORS_ORIGINS = ["*"]

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Secure settings for production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")

    # Production database (PostgreSQL recommended)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///movie_genie.db'

    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Production logging
    LOG_LEVEL = "WARNING"

    # Cache settings for production
    CACHE_TYPE = "redis" if os.environ.get('REDIS_URL') else "simple"
    CACHE_REDIS_URL = os.environ.get('REDIS_URL')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

    # Use in-memory database for tests
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

    # Disable CSRF protection in tests
    WTF_CSRF_ENABLED = False

# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on FLASK_ENV environment variable"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig)