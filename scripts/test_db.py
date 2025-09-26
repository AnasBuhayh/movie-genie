from movie_genie.backend.app.database.database import DatabaseManager
from movie_genie.backend.app.database.models import User, Movie, Session, Interaction, UserEmbedding

def test_database():
    # Initialize database
    db = DatabaseManager()
    
    # Reset database to ensure clean state for testing
    db.reset_database()
    
    # Test session creation in the SAME database connection
    with db.get_session() as db_session:
        # Create test user
        test_user = User(email="test@example.com", password_hash="dummy_hash")
        db_session.add(test_user)
        db_session.flush()  # Get the ID without committing
        
        print(f"✅ Created user with ID: {test_user.id}")
        # db_session.commit() happens automatically in context manager
        
    print("🎉 Database test completed successfully!")

if __name__ == "__main__":
    test_database()