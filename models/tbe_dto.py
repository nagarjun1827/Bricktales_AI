from pydantic import BaseModel
from typing import Optional

class TBEProcessingStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    message: str
    progress: Optional[float] = None

class TBEProcessingResult(BaseModel):
    success: bool
    project_id: Optional[int] = None
    location_id: Optional[int] = None
    boq_id: Optional[int] = None
    total_items: Optional[int] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    message: Optional[str] = None