"""
PostgreSQL connection manager using SQLAlchemy.
"""
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, Session
from core.settings import settings

logger = logging.getLogger(__name__)

# Build database URL from settings
DATABASE_URL = (
    f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=pool.QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,   # Recycle connections after 1 hour
    echo=False,          # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db_session() -> Session:
    """
    Get a database session.
    
    Usage:
        session = get_db_session()
        try:
            # Use session
            result = session.query(Model).all()
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    Returns:
        SQLAlchemy Session object
    """
    return SessionLocal()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_context() as session:
            result = session.query(Model).all()
            # Automatically commits on success, rolls back on exception
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        session.close()


def close_db_connections():
    """Close all database connections in the pool."""
    engine.dispose()
    logger.info("Database connections closed")