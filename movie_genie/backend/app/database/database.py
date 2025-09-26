from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager
import os

# This is our base class that all database models will inherit from
Base = declarative_base()

class DatabaseManager:
    """
    Manages database connections and sessions
    
    Why we need this:
    - Handles connection pooling efficiently
    - Provides thread-safe session management
    - Makes testing easier with different database URLs
    """
    def __init__(self, database_url: str = None):
        # Default to SQLite for development, easy to switch to PostgreSQL later
        self.database_url = database_url or "sqlite:///./movie_genie.db"
        
        # Create engine with connection pooling
        # echo=True shows SQL queries (helpful for debugging)
        self.engine = create_engine(
            self.database_url, 
            echo=False,  # Set to True to see SQL queries
            pool_pre_ping=True  # Validates connections before use
        )
        
        # Session factory - creates new sessions when needed
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
    
    @contextmanager
    def get_session(self):
        """
        Provides a database session with automatic cleanup
        
        Why we need this:
        - Guarantees session is closed even if exceptions occur
        - Handles transaction rollback on errors
        - Prevents database connection leaks
        """
        session = self.SessionLocal()
        try:
            yield session  # This is where the calling code runs
            session.commit()  # Save changes if everything succeeded
        except Exception:
            session.rollback()  # Undo changes if something failed
            raise  # Re-raise the exception so caller knows it failed
        finally:
            session.close()  # Always close the session

    def create_all_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        print("✅ All database tables created successfully")
    
    def reset_database(self):
        """Drop and recreate all tables (development only!)"""
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        print("🔄 Database reset complete")