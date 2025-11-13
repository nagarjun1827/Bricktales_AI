"""
Gemini AI tools for BOQ data extraction.
"""
import os
import re
import json
import logging
import google.generativeai as genai
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"


def clean_json_response(text: str) -> str:
    """Remove code block markers from Gemini response."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# === Sheet Structure Analysis ===
class SheetStructureInput(BaseModel):
    """Input for sheet structure analysis."""
    sheet_data: str = Field(description="First 30 rows of sheet data")
    sheet_name: str = Field(description="Sheet name")


class AnalyzeSheetStructureTool(BaseTool):
    """Analyzes Excel sheet structure using Gemini AI."""
    
    name: str = "analyze_sheet_structure"
    description: str = "Analyzes sheet structure to identify headers and columns"
    args_schema: Type[BaseModel] = SheetStructureInput

    def _run(self, sheet_data: str, sheet_name: str) -> str:
        """Analyze sheet structure."""
        prompt = f"""Analyze this Excel sheet structure.

Sheet: {sheet_name}
Data:
{sheet_data}

Identify:
1. has_header: boolean
2. header_row: row index (0-based, -1 if none)
3. data_start_row: row index (0-based)
4. column_structure: array of {{position, type, description}}

Column types: item_code, description, unit, quantity, supply_rate, labour_rate, supply_amount, labour_amount, total_amount

Notes:
- supply_rate: Unit rate for supply/material (may be labeled as "rate", "supply rate", "material rate", "unit rate")
- labour_rate: Unit rate for labour (may be labeled as "labour rate", "labor rate", "labour unit rate")
- supply_amount: Total supply amount (quantity × supply_rate, may be labeled as "supply amount", "material amount")
- labour_amount: Total labour amount (quantity × labour_rate, may be labeled as "labour amount", "labor amount")
- total_amount: Grand total (supply_amount + labour_amount, may be labeled as "total", "amount", "total amount")

Return ONLY valid JSON:
{{
    "has_header": true/false,
    "header_row": number,
    "data_start_row": number,
    "column_structure": [
        {{"position": 0, "type": "item_code", "description": "Item numbers"}},
        ...
    ]
}}"""

        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return clean_json_response(response.text)
        except Exception as e:
            logger.error(f"Sheet structure analysis failed: {e}")
            return json.dumps({"error": f"Analysis failed: {str(e)}"})


# === Project Information Extraction ===
class ProjectInfoInput(BaseModel):
    """Input for project extraction."""
    text: str = Field(description="Text from BOQ file")


class ExtractProjectInfoTool(BaseTool):
    """Extracts project information using Gemini AI."""
    
    name: str = "extract_project_info"
    description: str = "Extracts project name, code, client, and dates"
    args_schema: Type[BaseModel] = ProjectInfoInput

    def _run(self, text: str) -> str:
        """Extract project information."""
        prompt = f"""Extract project information from this text.

Text:
{text}

Return JSON with:
- project_name: string
- project_code: string (or generate)
- client_name: string or null
- start_year: integer or null
- end_year: integer or null

Return ONLY valid JSON."""

        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return clean_json_response(response.text)
        except Exception as e:
            logger.error(f"Project info extraction failed: {e}")
            return json.dumps({"error": f"Extraction failed: {str(e)}"})


# === Location Information Extraction ===
class LocationInfoInput(BaseModel):
    """Input for location extraction."""
    text: str = Field(description="Text from BOQ file")


class ExtractLocationInfoTool(BaseTool):
    """Extracts location information using Gemini AI."""
    
    name: str = "extract_location_info"
    description: str = "Extracts location name, city, and state"
    args_schema: Type[BaseModel] = LocationInfoInput

    def _run(self, text: str) -> str:
        """Extract location information."""
        prompt = f"""Extract location from this text.

Text:
{text}

Return JSON with:
- location_name: string (or "Unknown")
- city: string or null
- state: string or null

Return ONLY valid JSON."""

        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return clean_json_response(response.text)
        except Exception as e:
            logger.error(f"Location info extraction failed: {e}")
            return json.dumps({"error": f"Extraction failed: {str(e)}"})