from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from models.tbe_dto import TBEProcessingStatus, TBEProcessingResult
from services.tbe_boq_processor import TBEBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
from core.settings import settings
from pathlib import Path
import uuid
import shutil

router = APIRouter(prefix="/tbe-boq", tags=["To-Be-Estimated BOQ Processing"])

UPLOAD_DIR = Path(settings.UPLOAD_DIR) / "tbe"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def background_tbe_process(task_id: str, file_path: str, uploaded_by: str):
    """Background task for processing TBE BOQ"""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing TBE BOQ file..."

        service = TBEBOQProcessor()
        result = service.process_file(file_path, uploaded_by)

        processing_tasks[task_id].update({
            "status": "completed" if result["success"] else "failed",
            "result": result,
            "message": result.get("message", "Processing completed") if result["success"] else f"Error: {result.get('error')}",
        })
    except Exception as e:
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": str(e)},
            "message": f"Error: {str(e)}",
        })


@router.post("/upload", response_model=TBEProcessingStatus, summary="Upload TBE BOQ file")
async def upload_tbe_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Excel file (.xlsx, .xls) containing BOQ without rates"),
    uploaded_by: str = Query(default="system", description="User who uploaded the file"),
):
    """
    Upload and process a To-Be-Estimated BOQ file.
    
    TBE BOQ files contain:
    - Item codes and descriptions
    - Quantities and units
    - **NO rates or pricing** (to be estimated later)
    
    The system will:
    1. Extract project and location information
    2. Parse BOQ items (quantities only)
    3. Store in to_be_estimated_boq_files and to_be_estimated_boq_items tables
    4. Return task ID for tracking progress
    
    Next steps after processing:
    - Use semantic search to find similar items from store_boq_items
    - Apply rates based on similar items
    - Convert to store_boq_items once rates are applied
    """
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400, 
            detail="Only Excel files (.xlsx, .xls) are supported"
        )

    task_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"

    create_task(task_id)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Start background processing
    background_tasks.add_task(background_tbe_process, task_id, str(file_path), uploaded_by)

    return TBEProcessingStatus(
        task_id=task_id,
        status="pending",
        message="TBE BOQ processing started in background"
    )


@router.get("/status/{task_id}", response_model=TBEProcessingStatus, summary="Get processing status")
async def get_tbe_status(task_id: str):
    """
    Get the status of a TBE BOQ processing task.
    
    Possible statuses:
    - **pending**: Task created, waiting to start
    - **processing**: Currently processing the file
    - **completed**: Successfully processed
    - **failed**: Processing failed (check result for error details)
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TBEProcessingStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
    )


@router.get("/result/{task_id}", response_model=TBEProcessingResult, summary="Get processing result")
async def get_tbe_result(task_id: str):
    """
    Get the detailed result of a completed TBE BOQ processing task.
    
    Returns:
    - project_id: Created project ID
    - location_id: Created location ID
    - boq_id: Created TBE BOQ file ID
    - total_items: Number of items extracted (quantity only, no pricing)
    - processing_time: Time taken to process
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Processing not complete yet")
    
    return TBEProcessingResult(**task["result"])


@router.get("/items/{boq_id}", summary="Get TBE BOQ items")
async def get_tbe_items(
    boq_id: int,
    limit: int = Query(default=100, ge=1, le=1000, description="Items per page"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip")
):
    """
    Retrieve items from a To-Be-Estimated BOQ with pagination.
    
    Use this endpoint to:
    - View extracted items (quantity only, no rates)
    - Prepare for rate estimation
    - Export to other systems
    
    Items will have:
    - item_code, item_description, unit_of_measurement, quantity
    - **NO pricing fields** (supply_unit_rate, labour_unit_rate are not in TBE)
    """
    try:
        service = TBEBOQProcessor()
        result = service.get_tbe_items(boq_id, limit, offset)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/{boq_id}", summary="Get TBE BOQ summary")
async def get_tbe_summary(boq_id: int):
    """
    Get summary statistics for a TBE BOQ.
    
    Returns:
    - Total number of items extracted
    - BOQ metadata
    
    Note: TBE BOQs don't have pricing, so no amount calculations are included.
    """
    try:
        service = TBEBOQProcessor()
        summary = service.repo.get_tbe_boq_summary(boq_id)
        
        return {
            "success": True,
            "boq_id": boq_id,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))