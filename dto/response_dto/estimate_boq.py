from pydantic import BaseModel
from typing import Optional, List

class TBEProcessingStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: Optional[float] = None
    current_step: Optional[str] = None

class ItemWithPrice(BaseModel):
    item_id: int
    item_code: Optional[str]
    description: str
    unit: str
    quantity: float
    estimated_supply_rate: Optional[float]
    estimated_labour_rate: Optional[float]
    estimated_supply_total: Optional[float]
    estimated_labour_total: Optional[float]
    estimated_total: Optional[float]
    pricing_source: Optional[str] = None  # NEW: Source information
    similarity_score: Optional[float] = None  # NEW: How similar the match was

class TBEProcessingResult(BaseModel):
    success: bool
    
    # TBE Processing Results
    project_id: Optional[int] = None
    estimate_project_id: Optional[int] = None
    location_id: Optional[int] = None
    boq_id: Optional[int] = None
    total_items: Optional[int] = None
    
    # Price Fetching Results
    items_with_prices: Optional[int] = None
    items_without_prices: Optional[int] = None
    
    # Aggregated Price Summary
    total_estimated_supply: Optional[float] = None
    total_estimated_labour: Optional[float] = None
    total_estimated_amount: Optional[float] = None
    
    # Detailed Items with Prices
    items: List[ItemWithPrice] = []
    
    # Excel Export fields
    excel_available: Optional[bool] = None
    excel_data: Optional[str] = None  # Base64 encoded Excel file
    excel_error: Optional[str] = None
    
    # Timing
    file_processing_time: Optional[float] = None
    embedding_generation_time: Optional[float] = None
    price_fetching_time: Optional[float] = None
    total_processing_time: Optional[float] = None
    
    # Error
    error: Optional[str] = None
    error_step: Optional[str] = None
    message: Optional[str] = None

class DeleteResponse(BaseModel):
    success: bool
    boq_id: int
    deleted_counts: dict
    message: str