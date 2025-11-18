from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from dto.request_dto.store_boq import StoreBOQURLRequest
from dto.response_dto.store_boq import ProcessingStatus, ProcessingResult, DeleteResponse
from services.store_boq import StoreBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
import uuid
import logging
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/store-boq", tags=["Store BOQ"])

async def process_boq_background(task_id: str, file_url: str, uploaded_by: str):
    """Background task for BOQ processing from URL (ASYNC)."""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing BOQ file from URL..."

        processor = StoreBOQProcessor()
        result = await processor.process_file_from_url(file_url, uploaded_by)

        processing_tasks[task_id].update({
            "status": "completed" if result["success"] else "failed",
            "result": result,
            "message": "Completed" if result["success"] else f"Error: {result.get('error')}",
        })
    except Exception as e:
        logger.error(f"Background processing failed: {e}", exc_info=True)
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": str(e)},
            "message": f"Error: {str(e)}",
        })


@router.post("/upload", response_model=ProcessingStatus, summary="Upload Store BOQ File via URL")
async def upload_store_boq_url(
    background_tasks: BackgroundTasks,
    request: StoreBOQURLRequest = Body(...)
):
    """Upload and process store BOQ file from a URL/link."""
    task_id = str(uuid.uuid4())
    create_task(task_id)
    
    processing_tasks[task_id]["status"] = "pending"
    processing_tasks[task_id]["message"] = "Starting processing from URL..."

    # Add async task to background
    background_tasks.add_task(process_boq_background, task_id, str(request.file_url), request.uploaded_by)

    return ProcessingStatus(
        task_id=task_id,
        status="pending",
        message="Store BOQ processing started with automatic embedding generation from URL."
    )


@router.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
    """Get processing status."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return ProcessingStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
    )


@router.get("/result/{task_id}", response_model=ProcessingResult)
async def get_result(task_id: str):
    """Get processing result."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Not complete")
    return ProcessingResult(**task["result"])


@router.delete("/delete/{boq_id}", response_model=DeleteResponse, summary="Delete Store BOQ")
async def delete_store_boq(boq_id: int):
    """Delete a Store BOQ file and all its related data from the database."""
    try:
        processor = StoreBOQProcessor()
        result = await processor.delete_boq(boq_id)
        
        if not result['success']:
            raise HTTPException(
                status_code=404 if 'not found' in result.get('error', '').lower() else 500,
                detail=result.get('error', 'Failed to delete Store BOQ')
            )
        
        return DeleteResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete Store BOQ: {str(e)}"
        )


@router.get("/info/{boq_id}", summary="Get Store BOQ Information")
async def get_store_boq_info(boq_id: int):
    """Get basic information about a Store BOQ file."""
    try:
        processor = StoreBOQProcessor()
        boq_info = await processor.get_boq_info(boq_id)
        
        if not boq_info:
            raise HTTPException(
                status_code=404,
                detail=f"Store BOQ with ID {boq_id} not found"
            )
        
        return {
            "success": True,
            "data": boq_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Store BOQ info: {str(e)}"
        )