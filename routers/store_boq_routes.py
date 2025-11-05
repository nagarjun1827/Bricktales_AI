"""
API routes for store BOQ processing.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel, HttpUrl
from models.dto import ProcessingStatus, ProcessingResult, DeleteResponse
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

@router.delete("/delete/{boq_id}", response_model=DeleteResponse, summary="Delete Store BOQ")
async def delete_store_boq(boq_id: int):
    """
    Delete a Store BOQ file and all its related data from the database.
    
    **This will delete:**
    1. All BOQ items associated with this file
    2. The BOQ file record
    3. Store project (if no other BOQs exist for it)
    4. Locations (if store project is deleted)
    5. Main project (if no other store/estimate projects exist)
    
    **Parameters:**
    - **boq_id**: The ID of the Store BOQ file to delete
    
    **Returns:**
    - Success status
    - Count of deleted records from each table
    - Confirmation message
    
    **Example Response:**
```json
    {
        "success": true,
        "boq_id": 123,
        "deleted_counts": {
            "boq_items": 150,
            "boq_file": 1,
            "locations": 2,
            "store_project": 1,
            "project": 1
        },
        "message": "Store BOQ 123 and related data deleted successfully"
    }
```
    
    **Note:** This operation cannot be undone. Make sure you want to permanently delete this data.
    """
    try:
        from services.store_boq_processor import StoreBOQProcessor
        
        processor = StoreBOQProcessor()
        
        # Check if BOQ exists
        boq_info = processor.repo.get_boq_info(boq_id)
        if not boq_info:
            raise HTTPException(
                status_code=404,
                detail=f"Store BOQ with ID {boq_id} not found"
            )
        
        # Delete BOQ and related data
        result = processor.repo.delete_boq_by_id(boq_id)
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Failed to delete Store BOQ')
            )
        
        return DeleteResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Delete Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete Store BOQ: {str(e)}"
        )


@router.get("/info/{boq_id}", summary="Get Store BOQ Information")
async def get_store_boq_info(boq_id: int):
    """
    Get basic information about a Store BOQ file.
    
    **Parameters:**
    - **boq_id**: The ID of the Store BOQ file
    
    **Returns:**
    - BOQ ID, file name, project details, creation date
    """
    try:
        from services.store_boq_processor import StoreBOQProcessor
        
        processor = StoreBOQProcessor()
        boq_info = processor.repo.get_boq_info(boq_id)
        
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Store BOQ info: {str(e)}"
        )