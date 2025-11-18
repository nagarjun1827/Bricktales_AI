# dto/request_dto/tender.py
from pydantic import BaseModel, HttpUrl

class IngestRequest(BaseModel):
    url: HttpUrl
    uploaded_by: str = "user"

class SummarizeRequest(BaseModel):
    explanation_level: str = "professional"

class QueryRequest(BaseModel):
    question: str
    explanation_level: str = "professional"