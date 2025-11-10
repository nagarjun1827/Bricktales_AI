from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from dto.request_dto.estimate_boq import TBEProcessingStatus, TBEProcessingResult, DeleteResponse
from services.estimate_boq import TBEBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
import uuid
import io
import base64
import pandas as pd

router = APIRouter(prefix="/estimate-boq", tags=["Estimate BOQ Processing"])


class EstimateBOQURLRequest(BaseModel):
    """Request model for Estimate BOQ URL upload"""
    file_url: HttpUrl
    uploaded_by: str = "system"
    min_similarity: float = 0.5
    export_excel: bool = True


def background_tbe_process(
    task_id: str, 
    file_url: str, 
    uploaded_by: str,
    min_similarity: float,
    export_excel: bool = True
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
            top_k=1,  # Always 1
            min_similarity=min_similarity
        )

        # Generate Excel if requested and processing successful
        if export_excel and result.get("success"):
            try:
                processing_tasks[task_id]["message"] = "Generating Excel export..."
                processing_tasks[task_id]["current_step"] = "exporting_excel"
                
                excel_data = _generate_excel_data(result)
                result["excel_data"] = excel_data
                result["excel_available"] = True
                
            except Exception as excel_error:
                print(f"⚠️ Excel generation failed: {excel_error}")
                result["excel_available"] = False
                result["excel_error"] = str(excel_error)
        else:
            result["excel_available"] = False

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


def _generate_excel_data(result: dict) -> str:
    """Generate Excel data from processing result as base64 string"""
    items = result.get("items", [])
    
    if not items:
        raise ValueError("No items found in result")
    
    # Prepare data for DataFrame
    data = []
    for item in items:
        # Convert Pydantic model to dict
        item_data = item.model_dump() if hasattr(item, 'model_dump') else item
        
        data.append({
            'Item ID': item_data.get('item_id', ''),
            'Item Code': item_data.get('item_code', ''),
            'Description': item_data.get('description', ''),
            'Unit': item_data.get('unit', ''),
            'Quantity': item_data.get('quantity', ''),
            'Estimated Supply Rate': item_data.get('estimated_supply_rate'),
            'Estimated Labour Rate': item_data.get('estimated_labour_rate'),
            'Estimated Supply Total': item_data.get('estimated_supply_total'),
            'Estimated Labour Total': item_data.get('estimated_labour_total'),
            'Estimated Total': item_data.get('estimated_total'),
            'Pricing Source': item_data.get('pricing_source', ''),
            'Similarity Score': item_data.get('similarity_score')
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create Excel file in memory
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Estimate BOQ', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Estimate BOQ']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        currency_format = workbook.add_format({
            'num_format': '₹#,##0.00',
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '0.00',
            'border': 1
        })
        
        text_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'vcenter',
            'text_wrap': True
        })
        
        percent_format = workbook.add_format({
            'num_format': '0.0%',
            'border': 1
        })
        
        # Set column widths and formats
        worksheet.set_column('A:A', 10, text_format)  # Item ID
        worksheet.set_column('B:B', 12, text_format)  # Item Code
        worksheet.set_column('C:C', 50, text_format)  # Description
        worksheet.set_column('D:D', 10, text_format)  # Unit
        worksheet.set_column('E:E', 12, number_format)  # Quantity
        worksheet.set_column('F:F', 20, currency_format)  # Estimated Supply Rate
        worksheet.set_column('G:G', 20, currency_format)  # Estimated Labour Rate
        worksheet.set_column('H:H', 20, currency_format)  # Estimated Supply Total
        worksheet.set_column('I:I', 20, currency_format)  # Estimated Labour Total
        worksheet.set_column('J:J', 20, currency_format)  # Estimated Total
        worksheet.set_column('K:K', 60, text_format)  # Pricing Source
        worksheet.set_column('L:L', 15, percent_format)  # Similarity Score
        
        # Format header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Set row height for header
        worksheet.set_row(0, 20)
    
    # Get Excel data as bytes
    excel_bytes = output.getvalue()
    output.close()
    
    # Encode to base64
    excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
    
    print(f"✓ Excel data generated ({len(excel_bytes)} bytes)")
    return excel_base64


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
    4. **Price Fetching**: Find best matching item (top 1) and use its rates directly
    5. **Excel Export** (Optional): Generate Excel file with results if export_excel=true
    6. **Return Results**: Complete BOQ with estimated prices and pricing source
    
    **Request Body:**
```json
        {
            "file_url": "https://example.com/estimate_boq.xlsx",
            "uploaded_by": "user",
            "min_similarity": 0.5,
            "export_excel": true
        }
```
    
    **Note:** System always finds the single best matching item (top_k = 1) and uses its rates directly
    
    **Price Fetching Parameters:**
    - **min_similarity**: Minimum similarity score (0.0-1.0, default: 0.5)
    - **export_excel**: Generate Excel file after processing (default: false)
    
    **Excel Export Columns:**
    - Item ID, Item Code, Description, Unit, Quantity
    - Estimated Supply Rate, Estimated Labour Rate
    - Estimated Supply Total, Estimated Labour Total, Estimated Total
    - **Pricing Source** (explains where the price came from)
    - **Similarity Score** (how similar the match was)
    
    **Returns:**
    - Task ID for tracking progress
    - Use `/status/{task_id}` to check progress
    - Use `/result/{task_id}` to get complete results with prices
    - If export_excel=true, use `/download-excel/{task_id}` to download the Excel file
    """
    task_id = str(uuid.uuid4())

    # Create task
    create_task(task_id)
    
    # Update status
    processing_tasks[task_id]["status"] = "pending"
    processing_tasks[task_id]["message"] = "Starting processing from URL..."

    # Start background processing (top_k is always 1)
    background_tasks.add_task(
        background_tbe_process, 
        task_id, 
        str(request.file_url), 
        request.uploaded_by,
        request.min_similarity,
        request.export_excel
    )

    message = "Estimate BOQ processing started with automatic price fetching from URL (best match only)."
    if request.export_excel:
        message += " Excel export will be generated after processing."

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
    - **fetching_prices**: Finding best matching item and prices
    - **exporting_excel**: Generating Excel file (if requested)
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
    - **Detailed Items**: Each item with estimated rates, totals, pricing source, and similarity score
    - **Excel Availability**: excel_available flag
    - **Timing**: Processing time for each step
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Processing not complete yet")
    
    return TBEProcessingResult(**task["result"])


@router.get("/download-excel/{task_id}", summary="Download Excel file")
async def download_excel(task_id: str):
    """
    Download the generated Excel file for a completed task.
    
    **Excel Columns:**
    - Item ID, Item Code, Description, Unit, Quantity
    - Estimated Supply Rate, Estimated Labour Rate
    - Estimated Supply Total, Estimated Labour Total, Estimated Total
    - **Pricing Source** (detailed explanation of where the price came from)
    - **Similarity Score** (how similar the matched item was)
    
    **Note:** Excel file is only available if export_excel=true was set during upload.
    
    **Returns:**
    - Excel file download if available
    - Error if Excel was not generated or task not found
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
        
        if not result.get("excel_available"):
            error_msg = "Excel file was not generated. Set export_excel=true in the upload request to generate Excel."
            if result.get("excel_error"):
                error_msg += f" Error: {result['excel_error']}"
            raise HTTPException(status_code=404, detail=error_msg)
        
        excel_base64 = result.get("excel_data")
        
        if not excel_base64:
            raise HTTPException(
                status_code=404, 
                detail="Excel data not found"
            )
        
        # Decode base64 to bytes
        excel_bytes = base64.b64decode(excel_base64)
        
        # Return Excel as streaming response
        boq_id = result.get('boq_id', 'unknown')
        filename = f"estimate_boq_{boq_id}.xlsx"
        
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Excel Download Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to download Excel: {str(e)}"
        )


@router.delete("/delete/{boq_id}", response_model=DeleteResponse, summary="Delete Estimate BOQ")
async def delete_estimate_boq(boq_id: int):
    """
    Delete an Estimate BOQ file and all its related data from the database.
    
    **This will delete:**
    1. All BOQ items associated with this file
    2. The BOQ file record
    3. Estimate project (if no other BOQs exist for it)
    4. Locations (if estimate project is deleted)
    5. Main project (if no other store/estimate projects exist)
    
    **Parameters:**
    - **boq_id**: The ID of the Estimate BOQ file to delete
    
    **Returns:**
    - Success status
    - Count of deleted records from each table
    - Confirmation message
    
    **Example Response:**
```json
    {
        "success": true,
        "boq_id": 456,
        "deleted_counts": {
            "boq_items": 200,
            "boq_file": 1,
            "locations": 1,
            "estimate_project": 1,
            "project": 1
        },
        "message": "Estimate BOQ 456 and related data deleted successfully"
    }
```
    
    **Note:** This operation cannot be undone. Make sure you want to permanently delete this data.
    """
    try:
        from services.estimate_boq import TBEBOQProcessor
        
        processor = TBEBOQProcessor()
        
        # Check if BOQ exists
        boq_info = processor.repo.get_boq_info(boq_id)
        if not boq_info:
            raise HTTPException(
                status_code=404,
                detail=f"Estimate BOQ with ID {boq_id} not found"
            )
        
        # Delete BOQ and related data
        result = processor.repo.delete_boq_by_id(boq_id)
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Failed to delete Estimate BOQ')
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
            detail=f"Failed to delete Estimate BOQ: {str(e)}"
        )


@router.get("/info/{boq_id}", summary="Get Estimate BOQ Information")
async def get_estimate_boq_info(boq_id: int):
    """
    Get basic information about an Estimate BOQ file.
    
    **Parameters:**
    - **boq_id**: The ID of the Estimate BOQ file
    
    **Returns:**
    - BOQ ID, file name, project details, creation date
    """
    try:
        from services.estimate_boq import TBEBOQProcessor
        
        processor = TBEBOQProcessor()
        boq_info = processor.repo.get_boq_info(boq_id)
        
        if not boq_info:
            raise HTTPException(
                status_code=404,
                detail=f"Estimate BOQ with ID {boq_id} not found"
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
            detail=f"Failed to get Estimate BOQ info: {str(e)}"
        )