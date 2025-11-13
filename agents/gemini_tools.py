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
        prompt = f"""Analyze this Excel sheet structure carefully.

Sheet: {sheet_name}
Data:
{sheet_data}

Identify:
1. has_header: boolean (true if there's a header row)
2. header_row: row index (0-based, -1 if none)
3. data_start_row: row index (0-based) where actual data begins
4. column_structure: array of {{position, type, description}}

Column types to identify:
- item_code: Item number/serial number/S.No
- description: Item description/particulars/scope of work
- unit: Unit of measurement (UOM)
- quantity: Quantity/Qty
- supply_rate: Supply rate/Material rate/Supply unit rate/Unit rate (₹ per unit for material/supply)
- labour_rate: Labour rate/Labor rate/Labour unit rate (₹ per unit for labour)
- supply_amount: Supply amount/Material amount (quantity × supply_rate)
- labour_amount: Labour amount/Labor amount (quantity × labour_rate)
- total_amount: Total/Grand total/Amount (supply_amount + labour_amount)

IMPORTANT NOTES:
- Look for columns with headers like "Rate", "Supply Rate", "Material Rate", "Unit Rate" → these are supply_rate
- Look for columns with headers like "Labour Rate", "Labor Rate" → these are labour_rate
- Look for columns with currency symbols (₹) and numeric values after quantity column
- If you see only one "Rate" column, classify it as supply_rate
- Check the actual data values, not just headers

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "has_header": true/false,
    "header_row": number,
    "data_start_row": number,
    "column_structure": [
        {{"position": 0, "type": "item_code", "description": "Item numbers"}},
        {{"position": 1, "type": "description", "description": "Item descriptions"}},
        ...
    ]
}}"""

        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            cleaned = clean_json_response(response.text)
            logger.debug(f"Structure analysis result: {cleaned[:500]}")
            return cleaned
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

    def _run(self, text: str) -> str:  # Parameter name MUST match Field name in ProjectInfoInput
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

    def _run(self, text: str) -> str:  # Parameter name MUST match Field name in LocationInfoInput
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