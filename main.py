"""
QuoCo BOQ Processing API
"""
from fastapi import FastAPI
from routers.tbe_boq_routes import router as tbe_boq_router
from routers.store_boq_routes import router as store_boq_router

app = FastAPI(
    title="QuoCo BOQ API",
    description="AI-powered BoQ estimation from URL",
    version="2.0.0",
)

app.include_router(store_boq_router)
app.include_router(tbe_boq_router)

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "QuoCo BOQ API v2.0",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "store_boq": "/store-boq/upload - Process BOQ with rates via URL (direct processing)",
            "estimate_boq": "/estimate-boq/upload - Process BOQ without rates and fetch prices automatically via URL (direct processing)"
        },
        "features": [
            "Direct URL processing (no file uploads required)",
            "Automatic embedding generation",
            "Intelligent price fetching",
            "Optional CSV export"
        ]
    }

@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}