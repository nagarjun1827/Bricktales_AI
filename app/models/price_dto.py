from pydantic import BaseModel, Field
from typing import List, Optional

class PriceFetchRequest(BaseModel):
    boq_id: int = Field(..., description="BOQ ID to fetch prices for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of similar items per line item")
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")

class PriceFetchStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    message: str
    progress: Optional[float] = None

class PriceStatistics(BaseModel):
    avg: float
    min: float
    max: float
    median: float
    count: int

class SimilarItemDetail(BaseModel):
    item_id: int
    item_code: Optional[str]
    description: str
    supply_rate: float
    labour_rate: Optional[float]
    similarity: float
    project_name: str
    file_name: str

class LineItemPriceRecommendation(BaseModel):
    line_item_id: int
    item_code: Optional[str]
    description: str
    unit: str
    quantity: float
    similar_items_found: int
    supply_rate_stats: Optional[PriceStatistics]
    labour_rate_stats: Optional[PriceStatistics]
    estimated_supply_total: Optional[float]
    estimated_labour_total: Optional[float]
    estimated_total: Optional[float]
    similar_items: List[SimilarItemDetail] = []

class PriceFetchResult(BaseModel):
    success: bool
    boq_id: Optional[int] = None
    total_items: Optional[int] = None
    items_with_recommendations: Optional[int] = None
    items_without_recommendations: Optional[int] = None
    processing_time: Optional[float] = None
    recommendations: List[LineItemPriceRecommendation] = []
    error: Optional[str] = None