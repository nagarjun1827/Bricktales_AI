from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from dto.request_dto.estimate_boq import EstimateBOQURLRequest
from dto.response_dto.estimate_boq import TBEProcessingStatus, TBEProcessingResult, DeleteResponse
from services.estimate_boq import TBEBOQProcessor
from tasks.background_tasks import create_task, get_task, processing_tasks
import uuid
import io
import base64
import pandas as pd
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/estimate-boq", tags=["Estimate BOQ Processing"])

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
                logger.warning(f"Excel generation failed: {excel_error}")
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
        logger.error(f"Background processing failed: {e}", exc_info=True)
        processing_tasks[task_id].update({
            "status": "failed",
            "result": {"success": False, "error": str(e)},
            "message": f"Error: {str(e)}",
            "current_step": "failed"
        })


def _generate_excel_data(result: dict) -> str:
    """Generate Excel data from processing result as base64 string with Summary sheet"""
    items = result.get("items", [])
    
    if not items:
        raise ValueError("No items found in result")
    
    # Prepare data for Items DataFrame
    items_data = []
    for item in items:
        # Convert Pydantic model to dict
        item_data = item.model_dump() if hasattr(item, 'model_dump') else item
        
        items_data.append({
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
    
    # Create Items DataFrame
    items_df = pd.DataFrame(items_data)
    
    # Prepare Summary data
    summary_data = {
        'Metric': [
            'Total Items',
            'Items with Prices',
            'Items without Prices',
            '',
            'Total Estimated Supply Amount',
            'Total Estimated Labour Amount',
            'Total Estimated Amount',
            '',
            'File Processing Time (seconds)',
            'Embedding Generation Time (seconds)',
            'Price Fetching Time (seconds)',
            'Total Processing Time (seconds)',
        ],
        'Value': [
            result.get('total_items', 0),
            result.get('items_with_prices', 0),
            result.get('items_without_prices', 0),
            '',
            f"₹{result.get('total_estimated_supply', 0):,.2f}" if result.get('total_estimated_supply') else '₹0.00',
            f"₹{result.get('total_estimated_labour', 0):,.2f}" if result.get('total_estimated_labour') else '₹0.00',
            f"₹{result.get('total_estimated_amount', 0):,.2f}" if result.get('total_estimated_amount') else '₹0.00',
            '',
            f"{result.get('file_processing_time', 0):.2f}" if result.get('file_processing_time') else '0.00',
            f"{result.get('embedding_generation_time', 0):.2f}" if result.get('embedding_generation_time') else '0.00',
            f"{result.get('price_fetching_time', 0):.2f}" if result.get('price_fetching_time') else '0.00',
            f"{result.get('total_processing_time', 0):.2f}" if result.get('total_processing_time') else '0.00',
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create Excel file in memory with multiple sheets
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write Summary sheet first
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Write Items sheet
        items_df.to_excel(writer, sheet_name='Estimate BOQ', index=False)
        
        # Get workbook and worksheets
        workbook = writer.book
        summary_sheet = writer.sheets['Summary']
        items_sheet = writer.sheets['Estimate BOQ']
        
        # === SUMMARY SHEET FORMATTING ===
        
        # Define formats for summary
        summary_header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#2E75B6',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        summary_section_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D9E1F2',
            'border': 1,
            'align': 'left'
        })
        
        summary_currency_format = workbook.add_format({
            'align': 'right',
            'border': 1,
            'font_size': 11
        })
        
        summary_number_format = workbook.add_format({
            'align': 'right',
            'border': 1
        })
        
        # Set column widths for summary
        summary_sheet.set_column('A:A', 40)  # Metric column
        summary_sheet.set_column('B:B', 30)  # Value column
        
        # Format header row
        for col_num, value in enumerate(summary_df.columns.values):
            summary_sheet.write(0, col_num, value, summary_header_format)
        
        # Format rows with different styles
        for row_num in range(len(summary_df)):
            actual_row = row_num + 1  # +1 because header is row 0
            metric = summary_df.iloc[row_num]['Metric']
            value = summary_df.iloc[row_num]['Value']
            
            # Section headers (rows with specific metrics)
            if metric in ['Total Items', 'Total Estimated Supply Amount', 'File Processing Time (seconds)']:
                summary_sheet.write(actual_row, 0, metric, summary_section_format)
                summary_sheet.write(actual_row, 1, value, summary_number_format if not isinstance(value, str) or not value.startswith('₹') else summary_currency_format)
            # Empty rows
            elif metric == '':
                summary_sheet.write(actual_row, 0, '', workbook.add_format({'border': 0}))
                summary_sheet.write(actual_row, 1, '', workbook.add_format({'border': 0}))
            # Regular data rows
            else:
                summary_sheet.write(actual_row, 0, metric, workbook.add_format({'border': 1}))
                summary_sheet.write(actual_row, 1, value, summary_number_format if not isinstance(value, str) or not value.startswith('₹') else summary_currency_format)
        
        # Add title at the top
        summary_sheet.merge_range('A1:B1', 'BOQ ESTIMATION SUMMARY', workbook.add_format({
            'bold': True,
            'font_size': 14,
            'bg_color': '#2E75B6',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        }))
        
        # Shift data down by 2 rows to make room for title
        summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=2)
        
        # Re-apply formatting after writing data with offset
        for col_num, value in enumerate(summary_df.columns.values):
            summary_sheet.write(2, col_num, value, summary_header_format)
        
        for row_num in range(len(summary_df)):
            actual_row = row_num + 3  # +3 because title + header
            metric = summary_df.iloc[row_num]['Metric']
            value = summary_df.iloc[row_num]['Value']
            
            if metric in ['Total Items', 'Total Estimated Supply Amount', 'File Processing Time (seconds)']:
                summary_sheet.write(actual_row, 0, metric, summary_section_format)
                summary_sheet.write(actual_row, 1, value, summary_number_format if not isinstance(value, str) or not value.startswith('₹') else summary_currency_format)
            elif metric == '':
                summary_sheet.write(actual_row, 0, '', workbook.add_format({'border': 0}))
                summary_sheet.write(actual_row, 1, '', workbook.add_format({'border': 0}))
            else:
                summary_sheet.write(actual_row, 0, metric, workbook.add_format({'border': 1}))
                summary_sheet.write(actual_row, 1, value, summary_number_format if not isinstance(value, str) or not value.startswith('₹') else summary_currency_format)
        
        # === ITEMS SHEET FORMATTING ===
        
        # Define formats for items
        items_header_format = workbook.add_format({
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
        
        # Set column widths and formats for items
        items_sheet.set_column('A:A', 10, text_format)  # Item ID
        items_sheet.set_column('B:B', 12, text_format)  # Item Code
        items_sheet.set_column('C:C', 50, text_format)  # Description
        items_sheet.set_column('D:D', 10, text_format)  # Unit
        items_sheet.set_column('E:E', 12, number_format)  # Quantity
        items_sheet.set_column('F:F', 20, currency_format)  # Estimated Supply Rate
        items_sheet.set_column('G:G', 20, currency_format)  # Estimated Labour Rate
        items_sheet.set_column('H:H', 20, currency_format)  # Estimated Supply Total
        items_sheet.set_column('I:I', 20, currency_format)  # Estimated Labour Total
        items_sheet.set_column('J:J', 20, currency_format)  # Estimated Total
        items_sheet.set_column('K:K', 60, text_format)  # Pricing Source
        items_sheet.set_column('L:L', 15, percent_format)  # Similarity Score
        
        # Format header row for items
        for col_num, value in enumerate(items_df.columns.values):
            items_sheet.write(0, col_num, value, items_header_format)
        
        # Set row height for header
        items_sheet.set_row(0, 20)
    
    # Get Excel data as bytes
    excel_bytes = output.getvalue()
    output.close()
    
    # Encode to base64
    excel_base64 = base64.b64encode(excel_bytes).decode('utf-8')
    
    logger.info(f"Excel data generated with Summary sheet ({len(excel_bytes)} bytes)")
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
    5. **Excel Export** (Optional): Generate Excel file with results and summary if export_excel=true
    6. **Return Results**: Complete BOQ with estimated prices and pricing source
    
    **Request Body:**
```json
        {
            "file_url": "http://35.200.254.142:3000/uploads/1/d0746c9a18f05c96ce5ebee135afd362.xlsx",
            "uploaded_by": "user",
            "min_similarity": 0.5,
            "export_excel": true
        }
```
    
    **Note:** System always finds the single best matching item (top_k = 1) and uses its rates directly
    
    **Price Fetching Parameters:**
    - **min_similarity**: Minimum similarity score (0.0-1.0, default: 0.5)
    - **export_excel**: Generate Excel file after processing (default: false)
    
    **Excel Export:**
    - **Summary Sheet**: Overall statistics, totals, and processing time
    - **Estimate BOQ Sheet**: Detailed items with pricing
    
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
        message += " Excel export with summary will be generated after processing."

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
    - **exporting_excel**: Generating Excel file with summary (if requested)
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


@router.get("/download-excel/{task_id}", summary="Download Excel file with Summary")
async def download_excel(task_id: str):
    """
    Download the generated Excel file for a completed task.
    
    **Excel Contains:**
    
    **Sheet 1: Summary**
    - Total Items
    - Items with/without Prices
    - Total Estimated Supply Amount
    - Total Estimated Labour Amount
    - Total Estimated Amount
    - Processing Time Breakdown
    
    **Sheet 2: Estimate BOQ**
    - Item ID, Code, Description, Unit, Quantity
    - Estimated Supply Rate & Total
    - Estimated Labour Rate & Total
    - Estimated Total (supply + labour)
    - Pricing Source (detailed explanation)
    - Similarity Score (match quality)
    
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
        filename = f"estimate_boq_{boq_id}_with_summary.xlsx"
        
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Excel download error: {e}", exc_info=True)
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
        logger.error(f"Delete error: {e}", exc_info=True)
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