from pydantic import BaseModel
from typing import Optional, List

class TBEProcessingStatus(BaseModel):
    task_id: str
    status: str  # pending, processing_file, generating_embeddings, fetching_prices, completed, failed
    message: str
    progress: Optional[float] = None
    current_step: Optional[str] = None

class PriceStatistics(BaseModel):
    avg: float
    min: float
    max: float
    median: float
    count: int

class ItemWithPrice(BaseModel):
    item_id: int
    item_code: Optional[str]
    description: str
    unit: str
    quantity: float
    similar_items_found: int
    supply_rate_stats: Optional[PriceStatistics]
    labour_rate_stats: Optional[PriceStatistics]
    estimated_supply_rate: Optional[float]
    estimated_labour_rate: Optional[float]
    estimated_supply_total: Optional[float]
    estimated_labour_total: Optional[float]
    estimated_total: Optional[float]

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
    
    # Timing
    file_processing_time: Optional[float] = None
    embedding_generation_time: Optional[float] = None
    price_fetching_time: Optional[float] = None
    total_processing_time: Optional[float] = None
    
    # Error
    error: Optional[str] = None
    error_step: Optional[str] = None
    message: Optional[str] = None