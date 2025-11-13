from pydantic import BaseModel, HttpUrl

class EstimateBOQURLRequest(BaseModel):
    file_url: HttpUrl
    uploaded_by: str = "user"
    min_similarity: float = 0.5
    export_excel: bool = True