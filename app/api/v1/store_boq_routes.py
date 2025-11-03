"""
API routes for store BOQ processing.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from app.models.dto import ProcessingStatus, ProcessingResult
from app.services.store_boq_processor import StoreBOQProcessor
from app.tasks.background_tasks import create_task, get_task, processing_tasks
from app.core.settings import settings
from pathlib import Path
import uuid
import shutil

router = APIRouter(prefix="/store-boq", tags=["Store BOQ Processing"])

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)


def process_boq_background(task_id: str, file_path: str, uploaded_by: str):
    """Background task for BOQ processing."""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing BOQ file..."

        processor = StoreBOQProcessor()
        result = processor.process_file(file_path, uploaded_by)

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


@router.post("/upload", response_model=ProcessingStatus)
async def upload_store_boq(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    uploaded_by: str = Form(default="system")
):
    """
    Upload and process store BOQ file (with rates and pricing).
    
    Automatically generates semantic embeddings for all items.
    """
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files allowed")

    task_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"

    create_task(task_id)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_boq_background, task_id, str(file_path), uploaded_by)

    return ProcessingStatus(
        task_id=task_id,
        status="pending",
        message="Processing started with automatic embedding generation"
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