from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from models.price_dto import (
    PriceFetchRequest,
    PriceFetchStatus,
    PriceFetchResult,
    LineItemPriceRecommendation
)
from services.price_fetcher import PriceFetcher
from tasks.background_tasks import create_task, get_task, processing_tasks
import uuid
import csv
import io

router = APIRouter(prefix="/prices", tags=["Price Fetching"])


def background_price_fetch(task_id: str, boq_id: int, top_k: int, min_similarity: float):
    """Background task for fetching prices"""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Fetching prices..."

        service = PriceFetcher()
        result = service.fetch_prices_for_boq(
            boq_id=boq_id,
            top_k=top_k,
            min_similarity=min_similarity
        )

        processing_tasks[task_id].update({
            "status": "completed" if result["success"] else "failed",
            "result": result,
            "message": "Price fetching completed" if result["success"] else f"Error: {result.get('error')}",
        })
    except Exception as e:
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": str(e)},
            "message": f"Error: {str(e)}",
        })


@router.post("/fetch", response_model=PriceFetchStatus, summary="Fetch prices for BOQ")
async def fetch_prices(
    request: PriceFetchRequest,
    background_tasks: BackgroundTasks
):
    """
    Fetch price recommendations for all items in a BOQ.
    
    This endpoint uses semantic similarity to find similar items from historical BOQs
    and provides price recommendations based on those similar items.
    
    **Process:**
    1. For each line item in the target BOQ
    2. Generate embedding for item description
    3. Find top-k similar items from store_boq_items with matching unit
    4. Calculate price statistics (avg, min, max, median)
    5. Return price recommendations
    
    **Parameters:**
    - **boq_id**: BOQ to fetch prices for
    - **top_k**: Number of similar items to consider (1-20)
    - **min_similarity**: Minimum similarity threshold (0.0-1.0)
    
    **Returns:** Task ID for tracking progress
    """
    task_id = str(uuid.uuid4())
    create_task(task_id)

    background_tasks.add_task(
        background_price_fetch,
        task_id,
        request.boq_id,
        request.top_k,
        request.min_similarity
    )

    return PriceFetchStatus(
        task_id=task_id,
        status="pending",
        message="Price fetching started in background"
    )


@router.get("/status/{task_id}", response_model=PriceFetchStatus, summary="Get fetch status")
async def get_price_fetch_status(task_id: str):
    """
    Get the status of a price fetching task.
    
    **Statuses:**
    - **pending**: Task created, waiting to start
    - **processing**: Currently fetching prices
    - **completed**: Successfully completed
    - **failed**: Failed (check result for error)
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return PriceFetchStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
    )


@router.get("/result/{task_id}", response_model=PriceFetchResult, summary="Get fetch result")
async def get_price_fetch_result(task_id: str):
    """
    Get the result of a completed price fetching task.
    
    **Returns:**
    - Price recommendations for all line items
    - Statistics on items with/without recommendations
    - Detailed similar items for each line item
    - Processing time
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Task not complete yet")
    
    return PriceFetchResult(**task["result"])


@router.get("/export/{task_id}/csv", summary="Export results to CSV")
async def export_to_csv(task_id: str):
    """
    Export price recommendations to CSV file.
    
    **CSV Columns:**
    - Line Item ID, Item Code, Description, Unit, Quantity
    - Similar Items Found
    - Avg/Min/Max Supply Rate
    - Avg/Min/Max Labour Rate
    - Estimated Total (Supply)
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not complete yet")
    
    result = task["result"]
    recommendations = result.get("recommendations", [])
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Line Item ID', 'Item Code', 'Description', 'Unit', 'Quantity',
        'Similar Items Found',
        'Avg Supply Rate', 'Min Supply Rate', 'Max Supply Rate',
        'Avg Labour Rate', 'Min Labour Rate', 'Max Labour Rate',
        'Estimated Supply Total', 'Estimated Labour Total', 'Estimated Total'
    ])
    
    # Write data
    for rec in recommendations:
        supply_stats = rec.get('supply_rate_stats')
        labour_stats = rec.get('labour_rate_stats')
        
        writer.writerow([
            rec['line_item_id'],
            rec['item_code'] or '',
            rec['description'],
            rec['unit'],
            rec['quantity'],
            rec['similar_items_found'],
            f"{supply_stats['avg']:.2f}" if supply_stats else '',
            f"{supply_stats['min']:.2f}" if supply_stats else '',
            f"{supply_stats['max']:.2f}" if supply_stats else '',
            f"{labour_stats['avg']:.2f}" if labour_stats else '',
            f"{labour_stats['min']:.2f}" if labour_stats else '',
            f"{labour_stats['max']:.2f}" if labour_stats else '',
            f"{rec['estimated_supply_total']:.2f}" if rec.get('estimated_supply_total') else '',
            f"{rec['estimated_labour_total']:.2f}" if rec.get('estimated_labour_total') else '',
            f"{rec['estimated_total']:.2f}" if rec.get('estimated_total') else ''
        ])
    
    # Prepare response
    output.seek(0)
    boq_id = result.get('boq_id', 'unknown')
    filename = f"price_recommendations_boq_{boq_id}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/recommendations/{boq_id}", summary="Get price recommendations directly")
async def get_recommendations_sync(
    boq_id: int,
    top_k: int = Query(default=5, ge=1, le=20),
    min_similarity: float = Query(default=0.5, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """
    Get price recommendations synchronously (non-background).
    
    **Use this for:**
    - Small BOQs (< 100 items)
    - When you need immediate results
    - Testing and development
    
    **For large BOQs, use /fetch endpoint instead (background processing)**
    """
    try:
        service = PriceFetcher()
        result = service.fetch_prices_for_boq(
            boq_id=boq_id,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        # Apply pagination
        recommendations = result["recommendations"][offset:offset + limit]
        
        return {
            **result,
            "returned_items": len(recommendations),
            "recommendations": recommendations,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": (offset + len(recommendations)) < result["total_items"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))