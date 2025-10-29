from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from app.models.dto import ProcessingStatus, ProcessingResult
from app.services.boq_service import LangChainBOQService
from app.tasks.background import create_task, get_task, processing_tasks
from app.core.config import settings
from pathlib import Path
import uuid
import shutil

router = APIRouter(prefix="/boq", tags=["Embedding"])

@router.post("/embedding", response_model=ProcessingStatus)
async def embedding():
    pass