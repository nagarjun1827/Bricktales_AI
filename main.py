"""
QuoCo BOQ Processing API
"""
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from routers.estimate_boq import router as estimate_boq_router
from routers.store_boq import router as store_boq_router
from connections.db_init import init_db, check_db_connection
from connections.postgres_connection import close_db_connections

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting QuoCo BOQ API...")
    try:
        init_db(create_tables=False)  # Set to True to auto-create tables
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down QuoCo BOQ API...")
    close_db_connections()
    logger.info("Database connections closed")


app = FastAPI(
    title="QuoCo BOQ API",
    description="AI-powered BoQ estimation from URL",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(store_boq_router)
app.include_router(estimate_boq_router)

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "QuoCo BOQ API v2.0",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "store_boq": {
                "upload": "/store-boq/upload - Process BOQ with rates via URL",
                "status": "/store-boq/status/{task_id} - Check processing status",
                "result": "/store-boq/result/{task_id} - Get processing results",
                "info": "/store-boq/info/{boq_id} - Get BOQ information",
                "delete": "/store-boq/delete/{boq_id} - Delete BOQ and related data"
            },
            "estimate_boq": {
                "upload": "/estimate-boq/upload - Process BOQ without rates and fetch prices via URL",
                "status": "/estimate-boq/status/{task_id} - Check processing status",
                "result": "/estimate-boq/result/{task_id} - Get processing results",
                "download": "/estimate-boq/download-excel/{task_id} - Download Excel with estimates",
                "info": "/estimate-boq/info/{boq_id} - Get BOQ information",
                "delete": "/estimate-boq/delete/{boq_id} - Delete BOQ and related data"
            }
        },
        "features": [
            "Direct URL processing (no file uploads required)",
            "Automatic embedding generation",
            "Intelligent price fetching",
            "Excel export with pricing source",
            "Delete BOQ with cascade cleanup",
            "SQLAlchemy ORM with connection pooling"
        ]
    }

@app.get("/health")
def health():
    """Health check."""
    db_status = check_db_connection()
    return {
        "status": "healthy" if db_status else "degraded",
        "database": "connected" if db_status else "disconnected"
    }