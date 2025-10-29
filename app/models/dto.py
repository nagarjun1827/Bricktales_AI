from pydantic import BaseModel
from typing import Optional

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    message: str

class ProcessingResult(BaseModel):
    success: bool
    project_id: Optional[int] = None
    boq_id: Optional[int] = None
    total_items: Optional[int] = None
    total_supply: Optional[float] = None
    total_labour: Optional[float] = None
    total_amount: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None