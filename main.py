from fastapi import FastAPI
from app.api.v1.boq_router import router as boq_router
from app.api.v1.embedding_router import router as embedding_router
from app.api.v1.tbe_router import router as tbe_router

app = FastAPI(
    title="BrickTales - BOQ Processing API",
    description="AI-powered BOQ extraction and semantic search with TBE support",
    version="2.0.0",
)

# Include routers
app.include_router(boq_router)
app.include_router(embedding_router)
app.include_router(tbe_router)

@app.get("/")
def root() -> dict:
    return {
        "message": "BrickTales API is running!",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "store_boq": "/boq - Process BOQ files with rates (store_boq)",
            "tbe_boq": "/tbe-boq - Process BOQ files without rates (to-be-estimated)",
            "embeddings": "/embeddings - Semantic search and similarity"
        },
        "features": {
            "store_boq_processing": "Full BOQ with rates, quantities, and amounts",
            "tbe_boq_processing": "BOQ with quantities only (no rates) for estimation",
            "semantic_search": "Find similar items using AI embeddings",
            "rate_estimation": "Apply rates to TBE items based on similar store items"
        }
    }