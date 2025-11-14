import logging
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from core.settings import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Singleton database connection manager."""
    
    _sync_engine: Engine = None
    _async_engine: AsyncEngine = None
    _sync_session_factory = None
    _async_session_factory = None
    
    @classmethod
    def get_sync_engine(cls) -> Engine:
        """Get or create synchronous SQLAlchemy engine."""
        if cls._sync_engine is None:
            db_url = (
                f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            )
            
            cls._sync_engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            logger.info("Synchronous database engine created")
        
        return cls._sync_engine
    
    @classmethod
    def get_async_engine(cls) -> AsyncEngine:
        """Get or create asynchronous SQLAlchemy engine."""
        if cls._async_engine is None:
            # Use asyncpg driver for async operations
            db_url = (
                f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            )
            
            cls._async_engine = create_async_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            logger.info("Asynchronous database engine created")
        
        return cls._async_engine
    
    @classmethod
    def get_sync_session_factory(cls) -> sessionmaker:
        """Get or create synchronous session factory."""
        if cls._sync_session_factory is None:
            engine = cls.get_sync_engine()
            cls._sync_session_factory = sessionmaker(
                bind=engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            logger.info("Synchronous session factory created")
        
        return cls._sync_session_factory
    
    @classmethod
    def get_async_session_factory(cls) -> async_sessionmaker:
        """Get or create asynchronous session factory."""
        if cls._async_session_factory is None:
            engine = cls.get_async_engine()
            cls._async_session_factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            logger.info("Asynchronous session factory created")
        
        return cls._async_session_factory
    
    @classmethod
    async def close_async_engine(cls):
        """Close async engine properly."""
        if cls._async_engine:
            await cls._async_engine.dispose()
            cls._async_engine = None
            logger.info("Asynchronous engine disposed")
    
    @classmethod
    def close_sync_engine(cls):
        """Close sync engine."""
        if cls._sync_engine:
            cls._sync_engine.dispose()
            cls._sync_engine = None
            logger.info("Synchronous engine disposed")


# Convenience functions for getting sessions
def get_sync_session() -> Generator[Session, None, None]:
    """
    Get a synchronous database session.
    Use with context manager or dependency injection.
    
    Example:
        with next(get_sync_session()) as session:
            # Use session
    """
    SessionLocal = DatabaseConnection.get_sync_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an asynchronous database session.
    Use with async context manager or dependency injection.
    
    Example (FastAPI):
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_async_session)):
            # Use session
    """
    AsyncSessionLocal = DatabaseConnection.get_async_session_factory()
    async with AsyncSessionLocal() as session:
        yield session


# For FastAPI dependency injection
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for getting sync database session."""
    SessionLocal = DatabaseConnection.get_sync_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting async database session."""
    AsyncSessionLocal = DatabaseConnection.get_async_session_factory()
    async with AsyncSessionLocal() as session:
        yield session