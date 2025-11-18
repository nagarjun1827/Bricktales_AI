# routers/tender.py
"""
Tender document API router.
"""
import logging
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks, Body

from dto.request_dto.tender import IngestRequest, SummarizeRequest, QueryRequest
from dto.response_dto.tender import (
    IngestResponse, SummarizeResponse, QueryResponse, 
    TenderFileResponse, ProcessingStatus
)
from services.tender import TenderService
from tasks.background_tasks import create_task, get_task, processing_tasks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tender", tags=["Tender"])

tender_service = TenderService()


async def background_tender_process(task_id: str, file_url: str, uploaded_by: str):
    """Background task for tender processing."""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing tender document..."
        processing_tasks[task_id]["current_step"] = "ingesting"
        
        result = await tender_service.process_tender_document(file_url, uploaded_by)
        
        processing_tasks[task_id].update({
            "status": "completed" if result.get("tender_file_id", -1) > 0 else "failed",
            "result": result,
            "message": "Completed" if result.get("tender_file_id", -1) > 0 else f"Error: {result.get('error')}",
            "current_step": "completed" if result.get("tender_file_id", -1) > 0 else "failed"
        })
    except Exception as e:
        logger.error(f"Background processing failed: {e}", exc_info=True)
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"error": str(e), "tender_file_id": -1},
            "message": f"Error: {str(e)}",
            "current_step": "failed"
        })


@router.post("/ingest", response_model=ProcessingStatus, summary="Ingest Tender Document")
async def ingest_document(
    background_tasks: BackgroundTasks,
    request: IngestRequest = Body(...)
):
    """
    Ingest a tender document from URL.
    
    **Complete Workflow:**
    1. **Fetch PDF**: Download and extract text from PDF
    2. **Structure Extraction**: Extract metadata using AI
    3. **Chunking**: Break document into semantic chunks
    4. **Embedding Generation**: Create dense and sparse embeddings in parallel
    5. **Database Storage**: Store in PostgreSQL with pgvector
    
    **Request Body:**
    ```json
        {
            "url": "http://example.com/tender.pdf",
            "uploaded_by": "user123"
        }
    ```
    
    **Returns:**
    - Task ID for tracking progress
    - Use `/status/{task_id}` to check progress
    - Use `/result/{task_id}` to get results
    """
    task_id = str(uuid.uuid4())
    create_task(task_id)
    
    processing_tasks[task_id]["status"] = "pending"
    processing_tasks[task_id]["message"] = "Starting tender document processing..."
    
    background_tasks.add_task(background_tender_process, task_id, str(request.url), request.uploaded_by)
    
    return ProcessingStatus(
        task_id=task_id,
        status="pending",
        message="Tender document processing started",
        current_step="pending"
    )


@router.get("/status/{task_id}", response_model=ProcessingStatus, summary="Get Status")
async def get_status(task_id: str):
    """Get the status of a tender processing task."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return ProcessingStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        current_step=task.get("current_step")
    )


@router.get("/result/{task_id}", response_model=IngestResponse, summary="Get Result")
async def get_result(task_id: str):
    """Get the result of a completed tender processing task."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Processing not complete yet")
    
    result = task["result"]
    
    if result.get("tender_file_id", -1) <= 0:
        raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
    
    return IngestResponse(**result)


@router.post("/{tender_file_id}/summarize", response_model=SummarizeResponse, summary="Generate Summary")
async def summarize_document(tender_file_id: int, request: SummarizeRequest = Body(...)):
    """
    Generate summary for a tender document.
    
    **Parameters:**
    - **tender_file_id**: The ID of the tender file
    - **explanation_level**: "simple" or "professional" (default: "professional")
    
    **Returns:**
    - Summary text tailored to the explanation level
    """
    summary = await tender_service.generate_summary(tender_file_id, request.explanation_level)
    
    if "Error:" in summary or "not found" in summary.lower():
        raise HTTPException(status_code=404, detail=summary)
    
    return SummarizeResponse(summary=summary)


@router.post("/{tender_file_id}/query", response_model=QueryResponse, summary="Query Document")
async def query_document(tender_file_id: int, request: QueryRequest = Body(...)):
    """
    Ask a question about a tender document.
    
    **Parameters:**
    - **tender_file_id**: The ID of the tender file
    - **question**: The question to ask
    - **explanation_level**: "simple" or "professional" (default: "professional")
    
    **Returns:**
    - Answer based on relevant document sections
    """
    answer = await tender_service.answer_question(
        tender_file_id,
        request.question,
        request.explanation_level
    )
    
    if "Error:" in answer or "not found" in answer.lower():
        raise HTTPException(status_code=404, detail=answer)
    
    return QueryResponse(answer=answer)


@router.get("/{tender_file_id}", response_model=TenderFileResponse, summary="Get Tender File")
async def get_tender_file(tender_file_id: int):
    """
    Get tender file details.
    
    **Parameters:**
    - **tender_file_id**: The ID of the tender file
    
    **Returns:**
    - Tender file information including summaries
    """
    tender_file = await tender_service.get_tender_file_details(tender_file_id)
    
    if not tender_file:
        raise HTTPException(status_code=404, detail="Tender file not found")
    
    return TenderFileResponse(**tender_file)