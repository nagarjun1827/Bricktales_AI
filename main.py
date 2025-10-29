from fastapi import FastAPI
from app.api.v1.boq_router import router as boq_router
from app.api.v1.embedding import router as embedding_router

app = FastAPI(
    title="BrickTales - BOQ Processing API",
    description="AI-powered BOQ extraction",
    version="1.0.0",
)

app.include_router(boq_router)
app.include_router(embedding_router)

@app.get("/")
def root() -> dict:
    return {"message": "BrickTales API is running!"}