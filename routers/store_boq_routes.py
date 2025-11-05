"""
API routes for store BOQ processing.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel, HttpUrl
from models.dto import ProcessingStatus, ProcessingResult
from services.store_boq_processor import StoreBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
import uuid

router = APIRouter(prefix="/store-boq", tags=["Store BOQ Processing"])


class StoreBOQURLRequest(BaseModel):
    """Request model for Store BOQ URL upload"""
    file_url: HttpUrl
    uploaded_by: str = "system"


def process_boq_background(task_id: str, file_url: str, uploaded_by: str):
    """Background task for BOQ processing from URL."""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing BOQ file from URL..."

        processor = StoreBOQProcessor()
        result = processor.process_file_from_url(file_url, uploaded_by)

        processing_tasks[task_id].update({
            "status": "completed" if result["success"] else "failed",
            "result": result,
            "message": "Completed" if result["success"] else f"Error: {result.get('error')}",
        })
    except Exception as e:
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
    """
    Upload and process store BOQ file from a URL/link.
    
    **Workflow:**
    1. Read Excel file directly from provided URL
    2. Process BOQ and extract items with rates
    3. Automatically generate semantic embeddings
    4. Return complete results
    
    **Request Body:**
```json
{
        "file_url": "https://example.com/boq.xlsx",
        "uploaded_by": "user"
    }
    **Returns:**
    - Task ID for tracking progress
    - Use `/status/{task_id}` to check progress
    - Use `/result/{task_id}` to get final results
    """
    task_id = str(uuid.uuid4())

    # Create task
    create_task(task_id)
    
    processing_tasks[task_id]["status"] = "pending"
    processing_tasks[task_id]["message"] = "Starting processing from URL..."

    # Start background processing
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