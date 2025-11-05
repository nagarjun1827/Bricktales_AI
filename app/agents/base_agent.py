import os
import google.generativeai as genai
from core.config import settings

class BaseAgent:
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        self.model_name = model_name
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)