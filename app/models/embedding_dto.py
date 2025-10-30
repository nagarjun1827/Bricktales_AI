from pydantic import BaseModel, Field
from typing import Optional, List

class EmbeddingRequest(BaseModel):
    boq_id: int = Field(..., description="BOQ ID to generate embeddings for")
    model_name: str = Field(default="models/text-embedding-004", description="Embedding model to use")
    batch_size: int = Field(default=100, description="Batch size for processing")

class EmbeddingStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: Optional[float] = None

class EmbeddingResult(BaseModel):
    success: bool
    boq_id: Optional[int] = None
    total_items: Optional[int] = None
    embeddings_created: Optional[int] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class SimilarItemResponse(BaseModel):
    item_id: int
    item_code: Optional[str]
    item_description: str
    unit_of_measurement: str
    quantity: float
    supply_unit_rate: float
    similarity_score: float

class SimilaritySearchRequest(BaseModel):
    query: str = Field(..., description="Search query or item description")
    boq_id: Optional[int] = Field(None, description="Filter by BOQ ID")
    location_id: Optional[int] = Field(None, description="Filter by location ID")
    top_k: int = Field(default=10, description="Number of results to return")
    min_similarity: float = Field(default=0.5, description="Minimum similarity threshold")

class SimilaritySearchResponse(BaseModel):
    query: str
    total_results: int
    items: List[SimilarItemResponse]