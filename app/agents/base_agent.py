import os
import re
import google.generativeai as genai
from app.core.settings import settings

class BaseAgent:
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        self.model_name = model_name
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
    
    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response from Gemini"""
        text = text.strip()
        # Remove ```json ... ``` or ```
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()