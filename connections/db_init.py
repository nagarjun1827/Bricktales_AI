"""
Database initialization and health check utilities.
Supports both sync and async operations.
"""
import logging
import asyncio
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncEngine
from connections.postgres_connection import DatabaseConnection
from models.base import Base

logger = logging.getLogger(__name__)


def check_db_connection() -> bool:
    """
    Check if database connection is working (synchronous).
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        engine = DatabaseConnection.get_sync_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def check_db_connection_async() -> bool:
    """
    Check if database connection is working (asynchronous).
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        engine = DatabaseConnection.get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Async database connection successful")
        return True
    except Exception as e:
        logger.error(f"Async database connection failed: {e}")
        return False


def init_db(create_tables: bool = False):
    """
    Initialize database (synchronous).
    
    Args:
        create_tables: If True, create all tables defined in models
    """
    try:
        # Check connection
        if not check_db_connection():
            raise RuntimeError("Cannot connect to database")
        
        engine = DatabaseConnection.get_sync_engine()
        
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


async def init_db_async(create_tables: bool = False):
    """
    Initialize database (asynchronous).
    
    Args:
        create_tables: If True, create all tables defined in models
    """
    try:
        # Check connection
        if not await check_db_connection_async():
            raise RuntimeError("Cannot connect to database")
        
        engine = DatabaseConnection.get_async_engine()
        
        # Ensure pgvector extension exists
        logger.info("Ensuring pgvector extension...")
        async with engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
        logger.info("pgvector extension ready")
        
        # Create tables if requested
        if create_tables:
            logger.info("Creating database tables...")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Tables created successfully")
        
        # Log existing tables
        async with engine.connect() as conn:
            tables = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
            logger.info(f"Database has {len(tables)} tables")
        
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Async database initialization failed: {e}", exc_info=True)
        raise


def get_table_names() -> list:
    """Get list of all table names in the database (synchronous)."""
    engine = DatabaseConnection.get_sync_engine()
    inspector = inspect(engine)
    return inspector.get_table_names()


async def get_table_names_async() -> list:
    """Get list of all table names in the database (asynchronous)."""
    engine = DatabaseConnection.get_async_engine()
    async with engine.connect() as conn:
        tables = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
        return tables


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database (synchronous)."""
    engine = DatabaseConnection.get_sync_engine()
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


async def table_exists_async(table_name: str) -> bool:
    """Check if a table exists in the database (asynchronous)."""
    engine = DatabaseConnection.get_async_engine()
    async with engine.connect() as conn:
        tables = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
        return table_name in tables


# Convenience function to run async init from sync context
def init_db_with_async(create_tables: bool = False):
    """
    Initialize database using async engine from sync context.
    Useful for scripts and testing.
    """
    asyncio.run(init_db_async(create_tables))


# For backwards compatibility
def ensure_vector_extension():
    """Ensure pgvector extension is installed (synchronous)."""
    try:
        engine = DatabaseConnection.get_sync_engine()
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        logger.info("pgvector extension ensured")
    except Exception as e:
        logger.error(f"Failed to ensure pgvector extension: {e}")
        raise


async def ensure_vector_extension_async():
    """Ensure pgvector extension is installed (asynchronous)."""
    try:
        engine = DatabaseConnection.get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
        logger.info("pgvector extension ensured")
    except Exception as e:
        logger.error(f"Failed to ensure pgvector extension: {e}")
        raise