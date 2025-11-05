"""
API routes for store BOQ processing.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel, HttpUrl
from models.dto import ProcessingStatus, ProcessingResult
from services.store_boq_processor import StoreBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
from core.settings import settings
from pathlib import Path
import uuid
import shutil
import requests
import tempfile

router = APIRouter(prefix="/store-boq", tags=["Store BOQ Processing"])

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

class StoreBOQURLRequest(BaseModel):
    """Request model for Store BOQ URL upload"""
    file_url: HttpUrl
    uploaded_by: str = "system"


def download_file_from_url(url: str, destination: Path) -> bool:
    """Download file from URL to destination"""
    try:
        print(f"📥 Downloading file from: {url}")
        
        # Send GET request with stream=True for large files
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'excel' not in content_type.lower() and 'spreadsheet' not in content_type.lower():
            # Check if file extension is valid
            if not any(url.lower().endswith(ext) for ext in ['.xlsx', '.xls']):
                print(f"⚠️  Warning: Content-Type is '{content_type}', but proceeding anyway")
        
        # Write to file in chunks
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = destination.stat().st_size
        print(f"✓ Downloaded {file_size:,} bytes")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


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


@router.post("/upload", response_model=ProcessingStatus, summary="Upload Store BOQ File")
async def upload_store_boq_url(
    background_tasks: BackgroundTasks,
    request: StoreBOQURLRequest = Body(...)
):
    """
    Upload and process store BOQ file from a URL/link.
    
    **Workflow:**
    1. Download Excel file from provided URL
    2. Process BOQ and extract items with rates
    3. Automatically generate semantic embeddings
    4. Return complete results
    
    **Request Body:**
    ```json
        {
            "file_url": "https://example.com/boq.xlsx",
            "uploaded_by": "user"
        }
    ```
    
    **Returns:**
    - Task ID for tracking progress
    - Use `/status/{task_id}` to check progress
    - Use `/result/{task_id}` to get final results
    """
    task_id = str(uuid.uuid4())
    
    # Extract filename from URL or generate one
    url_path = Path(str(request.file_url))
    filename = url_path.name if url_path.suffix in ['.xlsx', '.xls'] else f"boq_{task_id}.xlsx"
    
    # Ensure proper extension
    if not filename.lower().endswith(('.xlsx', '.xls')):
        filename += '.xlsx'
    
    file_path = UPLOAD_DIR / f"{task_id}_{filename}"

    # Create task
    create_task(task_id)
    
    # Update status to downloading
    processing_tasks[task_id]["status"] = "downloading"
    processing_tasks[task_id]["message"] = "Downloading file from URL..."

    # Download file
    try:
        success = download_file_from_url(str(request.file_url), file_path)
        
        if not success:
            processing_tasks[task_id].update({
                "status": "failed",
                "result": {"success": False, "error": "Failed to download file from URL"},
                "message": "Failed to download file from URL",
            })
            return ProcessingStatus(
                task_id=task_id,
                status="failed",
                message="Failed to download file from URL"
            )
        
        # Verify it's an Excel file
        if not file_path.suffix.lower() in ['.xlsx', '.xls']:
            # Try to read as Excel anyway
            pass
        
    except Exception as e:
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": f"Download error: {str(e)}"},
            "message": f"Download error: {str(e)}",
        })
        return ProcessingStatus(
            task_id=task_id,
            status="failed",
            message=f"Download error: {str(e)}"
        )

    # Start background processing
    background_tasks.add_task(process_boq_background, task_id, str(file_path), request.uploaded_by)

    return ProcessingStatus(
        task_id=task_id,
        status="pending",
        message="File downloaded successfully. Processing started with automatic embedding generation."
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