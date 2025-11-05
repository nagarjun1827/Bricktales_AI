from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from models.tbe_dto import TBEProcessingStatus, TBEProcessingResult
from services.tbe_boq_processor import TBEBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
from pathlib import Path
import uuid
import csv
import io

router = APIRouter(prefix="/estimate-boq", tags=["Estimate BOQ Processing"])


class EstimateBOQURLRequest(BaseModel):
    """Request model for Estimate BOQ URL upload"""
    file_url: HttpUrl
    uploaded_by: str = "system"
    top_k: int = 1
    min_similarity: float = 0.5
    export_csv: bool = False


def background_tbe_process(
    task_id: str, 
    file_url: str, 
    uploaded_by: str,
    top_k: int,
    min_similarity: float,
    export_csv: bool = False
):
    """Background task for processing TBE BOQ with automatic price fetching"""
    try:
        # Update status
        processing_tasks[task_id]["status"] = "processing_file"
        processing_tasks[task_id]["message"] = "Processing TBE BOQ file from URL..."
        processing_tasks[task_id]["current_step"] = "file_processing"

        service = TBEBOQProcessor()
        result = service.process_file_from_url(
            file_url=file_url,
            uploaded_by=uploaded_by,
            top_k=top_k,
            min_similarity=min_similarity
        )

        # Generate CSV if requested and processing successful
        if export_csv and result.get("success"):
            try:
                processing_tasks[task_id]["message"] = "Generating CSV export..."
                processing_tasks[task_id]["current_step"] = "exporting_csv"
                
                csv_data = _generate_csv_data(result)
                result["csv_data"] = csv_data
                result["csv_available"] = True
                
            except Exception as csv_error:
                print(f"⚠️ CSV generation failed: {csv_error}")
                result["csv_available"] = False
                result["csv_error"] = str(csv_error)
        else:
            result["csv_available"] = False

        processing_tasks[task_id].update({
            "status": "completed" if result["success"] else "failed",
            "result": result,
            "message": result.get("message", "Processing completed") if result["success"] else f"Error: {result.get('error')}",
            "current_step": "completed" if result["success"] else "failed"
        })
    except Exception as e:
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": str(e)},
            "message": f"Error: {str(e)}",
            "current_step": "failed"
        })


def _generate_csv_data(result: dict) -> str:
    """Generate CSV data from processing result as string"""
    items = result.get("items", [])
    
    if not items:
        raise ValueError("No items found in result")
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Item ID', 'Item Code', 'Description', 'Unit', 'Quantity',
        'Similar Items Found',
        'Avg Supply Rate', 'Min Supply Rate', 'Max Supply Rate',
        'Avg Labour Rate', 'Min Labour Rate', 'Max Labour Rate',
        'Estimated Supply Rate', 'Estimated Labour Rate',
        'Estimated Supply Total', 'Estimated Labour Total', 'Estimated Total'
    ])
    
    # Helper function to format number
    def format_number(value):
        if value is None:
            return ''
        return f"{float(value):.2f}"
    
    # Write data
    for item in items:
        # Convert Pydantic model to dict
        item_data = item.model_dump() if hasattr(item, 'model_dump') else item
        
        supply_stats = item_data.get('supply_rate_stats')
        labour_stats = item_data.get('labour_rate_stats')
        
        writer.writerow([
            item_data.get('item_id', ''),
            item_data.get('item_code', ''),
            item_data.get('description', ''),
            item_data.get('unit', ''),
            item_data.get('quantity', ''),
            item_data.get('similar_items_found', 0),
            # Supply rate stats
            format_number(supply_stats.get('avg') if supply_stats else None),
            format_number(supply_stats.get('min') if supply_stats else None),
            format_number(supply_stats.get('max') if supply_stats else None),
            # Labour rate stats
            format_number(labour_stats.get('avg') if labour_stats else None),
            format_number(labour_stats.get('min') if labour_stats else None),
            format_number(labour_stats.get('max') if labour_stats else None),
            # Estimated rates
            format_number(item_data.get('estimated_supply_rate')),
            format_number(item_data.get('estimated_labour_rate')),
            # Estimated totals
            format_number(item_data.get('estimated_supply_total')),
            format_number(item_data.get('estimated_labour_total')),
            format_number(item_data.get('estimated_total'))
        ])
    
    csv_string = output.getvalue()
    output.close()
    
    print(f"✓ CSV data generated ({len(csv_string)} bytes)")
    return csv_string


@router.post("/upload", response_model=TBEProcessingStatus, summary="Upload Estimate BOQ File via URL")
async def upload_estimate_boq_url(
    background_tasks: BackgroundTasks,
    request: EstimateBOQURLRequest = Body(...)
):
    """
    Upload and process estimate BOQ file from a URL with automatic price fetching.
    
    **Complete Workflow:**
    1. **Process File from URL**: Read Excel file directly from URL
    2. **File Processing**: Extract items, project info, and location
    3. **Embedding Generation**: Generate semantic embeddings for all items
    4. **Price Fetching**: Find similar items and fetch price recommendations
    5. **CSV Export** (Optional): Generate CSV with results if export_csv=true
    6. **Return Results**: Complete BOQ with estimated prices
    
    **Request Body:**
```json
{
        "file_url": "https://example.com/estimate_boq.xlsx",
        "uploaded_by": "user",
        "top_k": 5,
        "min_similarity": 0.5,
        "export_csv": true
    }
    **Price Fetching Parameters:**
    - **top_k**: Number of similar items to consider (1-20, default: 5)
    - **min_similarity**: Minimum similarity score (0.0-1.0, default: 0.5)
    - **export_csv**: Generate CSV file after processing (default: false)
    
    **Returns:**
    - Task ID for tracking progress
    - Use `/status/{task_id}` to check progress
    - Use `/result/{task_id}` to get complete results with prices
    - If export_csv=true, use `/download-csv/{task_id}` to download the CSV file
    """
    task_id = str(uuid.uuid4())

    # Create task
    create_task(task_id)
    
    # Update status
    processing_tasks[task_id]["status"] = "pending"
    processing_tasks[task_id]["message"] = "Starting processing from URL..."

    # Start background processing
    background_tasks.add_task(
        background_tbe_process, 
        task_id, 
        str(request.file_url), 
        request.uploaded_by,
        request.top_k,
        request.min_similarity,
        request.export_csv
    )

    message = "Estimate BOQ processing started with automatic price fetching from URL."
    if request.export_csv:
        message += " CSV export will be generated after processing."

    return TBEProcessingStatus(
        task_id=task_id,
        status="pending",
        message=message,
        current_step="pending"
    )


@router.get("/status/{task_id}", response_model=TBEProcessingStatus, summary="Get status")
async def get_tbe_status(task_id: str):
    """
    Get the status of a TBE BOQ processing task.
    
    **Possible statuses:**
    - **pending**: Task created, waiting to start
    - **processing_file**: Extracting BOQ items from URL
    - **generating_embeddings**: Creating semantic embeddings
    - **fetching_prices**: Finding similar items and prices
    - **exporting_csv**: Generating CSV file (if requested)
    - **completed**: Successfully processed with prices
    - **failed**: Processing failed (check result for error details)
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TBEProcessingStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        current_step=task.get("current_step")
    )


@router.get("/result/{task_id}", response_model=TBEProcessingResult, summary="Get result")
async def get_tbe_result(task_id: str):
    """
    Get the detailed result of a completed TBE BOQ processing task.
    
    **Returns:**
    - **Project & BOQ Info**: project_id, estimate_project_id, location_id, boq_id
    - **Item Statistics**: total_items, items_with_prices, items_without_prices
    - **Price Estimates**: Aggregated totals and per-item recommendations
    - **Detailed Items**: Each item with:
      - Similar items found
      - Supply/Labour rate statistics (avg, min, max, median)
      - Estimated rates and totals
    - **CSV Availability**: csv_available flag
    - **Timing**: Processing time for each step
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Processing not complete yet")
    
    return TBEProcessingResult(**task["result"])


@router.get("/download-csv/{task_id}", summary="Download CSV file")
async def download_csv(task_id: str):
    """
    Download the generated CSV file for a completed task.
    
    **Note:** CSV file is only available if export_csv=true was set during upload.
    
    **Returns:**
    - CSV file download if available
    - Error if CSV was not generated or task not found
    """
    try:
        task = get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Task not completed yet. Current status: {task['status']}"
            )
        
        result = task["result"]
        
        if not result.get("csv_available"):
            error_msg = "CSV file was not generated. Set export_csv=true in the upload request to generate CSV."
            if result.get("csv_error"):
                error_msg += f" Error: {result['csv_error']}"
            raise HTTPException(status_code=404, detail=error_msg)
        
        csv_data = result.get("csv_data")
        
        if not csv_data:
            raise HTTPException(
                status_code=404, 
                detail="CSV data not found"
            )
        
        # Return CSV as streaming response
        boq_id = result.get('boq_id', 'unknown')
        filename = f"estimate_boq_{boq_id}.csv"
        
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"CSV Download Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to download CSV: {str(e)}"
        )