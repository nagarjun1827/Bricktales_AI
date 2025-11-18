# dto/response_dto/tender.py
from pydantic import BaseModel
from typing import Optional

class IngestResponse(BaseModel):
    message: str
    tender_file_id: int
    tender_id: int
    chunks_created: int
    processing_time: float

class SummarizeResponse(BaseModel):
    summary: str

class QueryResponse(BaseModel):
    answer: str

class TenderFileResponse(BaseModel):
    tender_file_id: int
    tender_id: int
    file_name: str
    file_path: str
    file_type: str
    summary: Optional[str] = None
    simple_summary: Optional[str] = None
    created_at: Optional[str] = None

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    message: str
    current_step: Optional[str] = None