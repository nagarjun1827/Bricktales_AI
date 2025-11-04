"""
BrickTales BOQ Processing API
"""
from fastapi import FastAPI
from routers.tbe_boq_routes import router as tbe_boq_router
from routers.store_boq_routes import router as store_boq_router

app = FastAPI(
    title="BrickTales BOQ API",
    description="AI-powered BOQ processing with automatic embeddings and price fetching",
    version="3.0.0",
)

app.include_router(store_boq_router)
app.include_router(tbe_boq_router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "BrickTales BOQ API v3.0",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "store_boq": "/store-boq - Process BOQ with rates",
            "tbe_boq": "/tbe-boq - Process BOQ without rates and fetch prices automatically"
        }
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}