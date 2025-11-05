"""
QuoCo BOQ Processing API
"""
from fastapi import FastAPI
from routers.tbe_boq_routes import router as tbe_boq_router
from routers.store_boq_routes import router as store_boq_router

app = FastAPI(
    title="QuoCo BOQ API",
    description="AI-powered BoQ estimation",
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
            "store_boq": "/store-boq - Process BOQ with rates",
            "estimate_boq": "/estimate-boq - Process BOQ without rates and fetch prices automatically"
        }
    }

@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}