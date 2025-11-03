"""
BrickTales BOQ Processing API
"""
from fastapi import FastAPI
from app.api.v1.store_boq_routes import router as store_boq_router
from app.api.v1.tbe_boq_routes import router as tbe_boq_router
from app.api.v1.price_routes import router as price_router

app = FastAPI(
    title="BrickTales BOQ API",
    description="AI-powered BOQ processing with automatic embeddings",
    version="2.1.0",
)

app.include_router(store_boq_router)
app.include_router(tbe_boq_router)
app.include_router(price_router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "BrickTales BOQ API v2.1",
        "version": "2.1.0",
        "docs": "/docs",
        "endpoints": {
            "store_boq": "/store-boq - Process BOQ with rates",
            "tbe_boq": "/tbe-boq - Process BOQ without rates",
            "prices": "/prices - Get price recommendations"
        }
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}