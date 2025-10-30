from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.embedding_dto import (
    EmbeddingRequest,
    EmbeddingStatus,
    EmbeddingResult,
    SimilaritySearchRequest,
    SimilaritySearchResponse
)
from app.services.embedding_service import EmbeddingService
from app.tasks.background import create_task, get_task, processing_tasks
import uuid

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])

def background_embedding_task(task_id: str, boq_id: int, model_name: str, batch_size: int):
    """Background task for generating embeddings"""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Generating embeddings..."

        service = EmbeddingService()
        result = service.generate_embeddings_for_boq(
            boq_id=boq_id,
            model_name=model_name,
            batch_size=batch_size
        )

        processing_tasks[task_id].update({
            "status": "completed" if result["success"] else "failed",
            "result": result,
            "message": "Embedding generation completed" if result["success"] else f"Error: {result.get('error')}",
        })
    except Exception as e:
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": str(e)},
            "message": f"Error: {str(e)}",
        })

@router.post("/initialize", summary="Initialize embedding database")
async def initialize_database():
    """
    Initialize database with pgvector extension.
    Run this once before generating embeddings for the first time.
    """
    try:
        service = EmbeddingService()
        service.initialize_database()
        return {"success": True, "message": "Database initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=EmbeddingStatus, summary="Generate embeddings for BOQ")
async def generate_embeddings(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate embeddings for all items in a BOQ file.
    This runs in the background and returns a task ID for tracking.
    
    - **boq_id**: ID of the BOQ to process
    - **model_name**: Embedding model (default: models/text-embedding-004)
    - **batch_size**: Number of items to process at once (default: 100)
    """
    task_id = str(uuid.uuid4())
    create_task(task_id)

    background_tasks.add_task(
        background_embedding_task,
        task_id,
        request.boq_id,
        request.model_name,
        request.batch_size
    )

    return EmbeddingStatus(
        task_id=task_id,
        status="pending",
        message="Embedding generation started in background"
    )

@router.post("/regenerate/{boq_id}", response_model=EmbeddingStatus, summary="Regenerate embeddings")
async def regenerate_embeddings(
    boq_id: int,
    background_tasks: BackgroundTasks,
    model_name: str = "models/text-embedding-004",
    batch_size: int = 100
):
    """
    Clear and regenerate all embeddings for a BOQ.
    Useful when you want to update embeddings with a different model or after data changes.
    
    - **boq_id**: ID of the BOQ to regenerate embeddings for
    - **model_name**: Embedding model to use
    - **batch_size**: Number of items to process at once
    """
    task_id = str(uuid.uuid4())
    create_task(task_id)

    def regenerate_task(task_id: str):
        try:
            processing_tasks[task_id]["status"] = "processing"
            processing_tasks[task_id]["message"] = "Regenerating embeddings..."

            service = EmbeddingService()
            result = service.regenerate_embeddings(boq_id, model_name)

            processing_tasks[task_id].update({
                "status": "completed" if result["success"] else "failed",
                "result": result,
                "message": "Regeneration completed" if result["success"] else f"Error: {result.get('error')}",
            })
        except Exception as e:
            processing_tasks[task_id].update({
                "status": "failed",
                "result": {"success": False, "error": str(e)},
                "message": f"Error: {str(e)}",
            })

    background_tasks.add_task(regenerate_task, task_id)

    return EmbeddingStatus(
        task_id=task_id,
        status="pending",
        message="Embedding regeneration started in background"
    )

@router.get("/status/{task_id}", response_model=EmbeddingStatus, summary="Get embedding task status")
async def get_embedding_status(task_id: str):
    """
    Get the status of an embedding generation task.
    
    - **task_id**: Task ID returned from generate or regenerate endpoint
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return EmbeddingStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
    )

@router.get("/result/{task_id}", response_model=EmbeddingResult, summary="Get embedding result")
async def get_embedding_result(task_id: str):
    """
    Get the result of a completed embedding generation task.
    
    - **task_id**: Task ID returned from generate or regenerate endpoint
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Task not complete yet")
    
    return EmbeddingResult(**task["result"])

@router.post("/search", response_model=SimilaritySearchResponse, summary="Search similar items")
async def search_similar_items(request: SimilaritySearchRequest):
    """
    Search for similar BOQ items using semantic similarity.
    Returns items ranked by cosine similarity to the query.
    
    - **query**: Search query or item description
    - **boq_id**: Optional - filter by specific BOQ
    - **location_id**: Optional - filter by specific location
    - **top_k**: Number of results to return (default: 10)
    - **min_similarity**: Minimum similarity threshold 0-1 (default: 0.5)
    """
    try:
        service = EmbeddingService()
        result = service.search_similar_items(
            query=request.query,
            top_k=request.top_k,
            boq_id=request.boq_id,
            location_id=request.location_id,
            min_similarity=request.min_similarity
        )
        
        return SimilaritySearchResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", summary="Get embedding statistics")
async def get_embedding_stats(boq_id: int = None):
    """
    Get statistics about embeddings.
    
    - **boq_id**: Optional - if provided, returns stats for that BOQ only. Otherwise returns global stats.
    """
    try:
        service = EmbeddingService()
        stats = service.get_embedding_statistics(boq_id)
        return {
            "success": True,
            "boq_id": boq_id,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))