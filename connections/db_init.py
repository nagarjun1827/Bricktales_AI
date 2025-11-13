"""
Database initialization and health check utilities.
"""
import logging
from sqlalchemy import text, inspect
from connections.postgres_connection import engine, get_db_session
from models.base import Base

logger = logging.getLogger(__name__)


def check_db_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def init_db(create_tables: bool = False):
    """
    Initialize database.
    
    Args:
        create_tables: If True, create all tables defined in models
    """
    try:
        # Check connection
        if not check_db_connection():
            raise RuntimeError("Cannot connect to database")
        
        # Ensure pgvector extension exists
        logger.info("Ensuring pgvector extension...")
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        logger.info("pgvector extension ready")
        
        # Create tables if requested
        if create_tables:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=engine)
            logger.info("Tables created successfully")
        
        # Log existing tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Database has {len(tables)} tables: {', '.join(tables[:10])}")
        
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise


def get_table_names() -> list:
    """Get list of all table names in the database."""
    inspector = inspect(engine)
    return inspector.get_table_names()


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()