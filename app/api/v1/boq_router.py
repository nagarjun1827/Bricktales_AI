from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from app.models.dto import ProcessingStatus, ProcessingResult
from app.services.boq_service import LangChainBOQService
from app.tasks.background import create_task, get_task, processing_tasks
from app.core.config import settings
from pathlib import Path
import uuid
import shutil

router = APIRouter(prefix="/boq", tags=["Storage BOQ Processing"])

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

def background_process(task_id: str, file_path: str, uploaded_by: str):
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing BOQ file..."

        service = LangChainBOQService()
        result = service.process_file(file_path, uploaded_by)

        processing_tasks[task_id].update(
            {
                "status": "completed" if result["success"] else "failed",
                "result": result,
                "message": "Processing completed" if result["success"] else f"Error: {result.get('error')}",
            }
        )
    except Exception as e:
        processing_tasks[task_id].update(
            {
                "status": "failed",
                "result": {"success": False, "error": str(e)},
                "message": f"Error: {str(e)}",
            }
        )

@router.post("/upload", response_model=ProcessingStatus)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    uploaded_by: str = "system",
):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")

    task_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"

    create_task(task_id)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(background_process, task_id, str(file_path), uploaded_by)

    return ProcessingStatus(
        task_id=task_id, status="pending", message="Processing started in background"
    )

@router.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
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
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Processing not complete yet")
    return ProcessingResult(**task["result"])