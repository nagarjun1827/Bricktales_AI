from pydantic import BaseModel, HttpUrl
from typing import Optional

class StoreBOQURLRequest(BaseModel):
    """Request model for Store BOQ URL upload"""
    file_url: HttpUrl
    uploaded_by: str = "system"

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: Optional[float] = None 

class ProcessingResult(BaseModel):
    success: bool
    project_id: Optional[int] = None
    boq_id: Optional[int] = None
    total_items: Optional[int] = None
    total_supply: Optional[float] = None
    total_labour: Optional[float] = None
    total_amount: Optional[float] = None
    embeddings_generated: Optional[bool] = None
    embeddings_count: Optional[int] = None
    embedding_time: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class DeleteResponse(BaseModel):
    success: bool
    boq_id: int
    deleted_counts: dict
    message: str