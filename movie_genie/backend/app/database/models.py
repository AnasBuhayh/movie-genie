from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from .database import Base

class User(Base):
    """User model for authentication and preferences"""
    __tablename__ = "users"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Authentication
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # User preferences
    preferences_text = Column(String, nullable=True)
    
    # ADD THESE MISSING FIELDS:
    account_status = Column(String, default="active")
    deleted_at = Column(DateTime, nullable=True)
    
    # Relationships (keep existing)
    sessions = relationship("Session", back_populates="user")
    interactions = relationship("Interaction", back_populates="user")
    embedding = relationship("UserEmbedding", back_populates="user", uselist=False)
    
    # Table optimization (keep existing)
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_created', 'created_at'),
    )


class Session(Base):
    """
    User session tracking for sequential modeling
    
    Business value: Understanding user behavior patterns within sessions
    helps us make better immediate recommendations
    """
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    start_time = Column(DateTime, default=datetime.utcnow, index=True)
    end_time = Column(DateTime, nullable=True)  # Null while session is active

    # Session context for better recommendations
    device_info = Column(JSON, nullable=True)

    # Relationships  
    user = relationship("User", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session")
    
    # Optimized indexes for common queries
    __table_args__ = (
        Index('idx_user_session_time', 'user_id', 'start_time'),  # User's session history
        Index('idx_active_sessions', 'end_time'),                 # Find active sessions (WHERE end_time IS NULL)
    )

class Interaction(Base):
    """
    User-movie interactions with session context
    
    This is our most important table - every recommendation algorithm
    depends on this interaction data
    """
    __tablename__ = "interactions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)

    # Rating can be explicit (1-5 stars) or implicit (derived from behavior)
    rating = Column(Float, nullable=True)

    interaction_type = Column(String, default="rating")

    # Critical for temporal modeling and sequential recommendations
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="interactions")
    session = relationship("Session", back_populates="interactions") 
    movie = relationship("Movie", back_populates="interactions")
    
    # Database indexes for fast queries
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),      # User activity timeline
        Index('idx_session_timestamp', 'session_id', 'timestamp'), # Session sequences  
        Index('idx_movie_timestamp', 'movie_id', 'timestamp'),    # Movie popularity over time
    )

class Movie(Base):
    """Movies with rich TMDB metadata and review stats"""
    __tablename__ = "movies"
    
    # Core identifiers
    id = Column(Integer, primary_key=True)  # MovieLens ID
    tmdb_id = Column(Integer, nullable=True, index=True)
    imdb_id = Column(String, nullable=True, index=True)
    
    # Basic metadata
    title = Column(String, nullable=False, index=True)
    year = Column(Integer, nullable=True, index=True)
    genres = Column(JSON, nullable=True)  # ["Action", "Sci-Fi"]
    
    # Rich features (JSON for flexibility)
    tmdb_features = Column(JSON, nullable=True)  # Cast, director, budget, etc.
    review_stats = Column(JSON, nullable=True)   # Sentiment, themes from IMDB
    
    # Relationships
    interactions = relationship("Interaction", back_populates="movie")
    
    # Indexes for search and filtering
    __table_args__ = (
        Index('idx_movie_title_year', 'title', 'year'),
        Index('idx_movie_genres', 'genres'),
    )


class UserEmbedding(Base):
    """Real-time user embeddings (separate for frequent updates)"""
    __tablename__ = "user_embeddings"
    
    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    embedding_vector = Column(JSON, nullable=False)  # [0.1, 0.5, -0.2, ...]
    last_updated = Column(DateTime, default=datetime.utcnow, index=True)
    model_version = Column(String, default="v1.0")
    
    # Relationship
    user = relationship("User", back_populates="embedding")