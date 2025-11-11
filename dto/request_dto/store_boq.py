from pydantic import BaseModel, HttpUrl

class StoreBOQURLRequest(BaseModel):
    file_url: HttpUrl
    uploaded_by: str = "user"
