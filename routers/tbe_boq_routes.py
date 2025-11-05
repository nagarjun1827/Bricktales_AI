from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import StreamingResponse
from models.tbe_dto import TBEProcessingStatus, TBEProcessingResult
from services.tbe_boq_processor import TBEBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
from core.settings import settings
from pathlib import Path
import uuid
import shutil
import csv
import io

router = APIRouter(prefix="/estimate-boq", tags=["Estimate BOQ Processing"])

UPLOAD_DIR = Path(settings.UPLOAD_DIR) / "tbe"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def background_tbe_process(
    task_id: str, 
    file_path: str, 
    uploaded_by: str,
    top_k: int,
    min_similarity: float
):
    """Background task for processing TBE BOQ with automatic price fetching"""
    try:
        # Update status
        processing_tasks[task_id]["status"] = "processing_file"
        processing_tasks[task_id]["message"] = "Processing TBE BOQ file..."
        processing_tasks[task_id]["current_step"] = "file_processing"

        service = TBEBOQProcessor()
        result = service.process_file(
            file_path=file_path,
            uploaded_by=uploaded_by,
            top_k=top_k,
            min_similarity=min_similarity
        )

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


@router.post("/upload", response_model=TBEProcessingStatus, summary="Upload TBE BOQ file with automatic price fetching")
async def upload_tbe_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Excel file (.xlsx, .xls) containing BOQ without rates"),
    uploaded_by: str = Form(default="system", description="User who uploaded the file"),
    top_k: int = Form(default=5, ge=1, le=20, description="Number of similar items to find per line item"),
    min_similarity: float = Form(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold for price matching"),
):
    """
    Upload and process a To-Be-Estimated BOQ file with automatic price fetching.
    
    **Complete Workflow:**
    1. **File Processing**: Extract items, project info, and location
    2. **Embedding Generation**: Generate semantic embeddings for all items
    3. **Price Fetching**: Find similar items and fetch price recommendations
    4. **Return Results**: Complete BOQ with estimated prices
    
    **TBE BOQ files contain:**
    - Item codes and descriptions
    - Quantities and units
    - **NO rates or pricing** (to be estimated automatically)
    
    **Price Fetching Parameters:**
    - **top_k**: Number of similar items to consider (1-20, default: 5)
    - **min_similarity**: Minimum similarity score (0.0-1.0, default: 0.5)
    
    **Returns:**
    - Task ID for tracking progress
    - Complete results with estimated prices when done
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
    background_tasks.add_task(
        background_tbe_process, 
        task_id, 
        str(file_path), 
        uploaded_by,
        top_k,
        min_similarity
    )

    return TBEProcessingStatus(
        task_id=task_id,
        status="pending",
        message="TBE BOQ processing started with automatic price fetching",
        current_step="pending"
    )


@router.get("/status/{task_id}", response_model=TBEProcessingStatus, summary="Get status")
async def get_tbe_status(task_id: str):
    """
    Get the status of a TBE BOQ processing task.
    
    **Possible statuses:**
    - **pending**: Task created, waiting to start
    - **processing_file**: Extracting BOQ items
    - **generating_embeddings**: Creating semantic embeddings
    - **fetching_prices**: Finding similar items and prices
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
    - **Timing**: Processing time for each step
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Processing not complete yet")
    
    return TBEProcessingResult(**task["result"])


@router.get("/items/{boq_id}", summary="Get TBE BOQ items (without prices)")
async def get_tbe_items(
    boq_id: int,
    limit: int = Query(default=100, ge=1, le=1000, description="Items per page"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip")
):
    """
    Retrieve items from a To-Be-Estimated BOQ (without price information).
    
    **Note:** This endpoint returns raw BOQ items without prices.
    For items with price estimates, use the /result/{task_id} endpoint after processing.
    
    Items will have:
    - item_code, item_description, unit_of_measurement, quantity
    - **NO pricing fields** (use /result for prices)
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
    
    **Note:** TBE BOQs don't have pricing until processed, so no amount calculations are included.
    For estimated amounts, use the /result/{task_id} endpoint.
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


@router.get("/export/{task_id}/csv", summary="Export results to CSV")
async def export_to_csv(task_id: str):
    """
    Export TBE BOQ items with price recommendations to CSV file.
    """
    try:
        task = get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] != "completed":
            raise HTTPException(status_code=400, detail="Task not complete yet")
        
        result = task["result"]
        items = result.get("items", [])
        
        if not items:
            raise HTTPException(status_code=404, detail="No items found in result")
        
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
        
        # Write data - convert each item to dict first
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
        
        # Prepare response
        output.seek(0)
        boq_id = result.get('boq_id', 'unknown')
        filename = f"tbe_boq_with_prices_{boq_id}.csv"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"CSV Export Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate CSV: {str(e)}"
        )