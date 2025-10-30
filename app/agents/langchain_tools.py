# app/agents/langchain_tools.py
import os
import re
import json
import google.generativeai as genai
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool  # Optional: only if used elsewhere
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API key globally (once)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)


def _clean_json_response(text: str) -> str:
    """
    Removes code block markers and extra whitespace from LLM response.
    """
    text = text.strip()
    # Remove ```json ... ``` or ```
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# === TOOL 1: Analyze Sheet Structure ===
class AnalyzeSheetStructureInput(BaseModel):
    sheet_data: str = Field(description="First 30 rows of sheet data as text")
    sheet_name: str = Field(description="Name of the sheet")


class AnalyzeSheetStructureTool(BaseTool):
    name: str = "analyze_sheet_structure"
    description: str = (
        "Analyzes Excel sheet structure to identify headers and data layout. "
        "Returns JSON with has_header, header_row, data_start_row, and column_structure."
    )
    args_schema: Type[BaseModel] = AnalyzeSheetStructureInput

    def _run(self, sheet_data: str, sheet_name: str) -> str:
        prompt = f"""You are a BOQ sheet structure analyzer. Analyze this Excel sheet.

Sheet: {sheet_name}
Data:
{sheet_data}

Determine:
1. has_header: Does this sheet have a header row? (true/false)
2. header_row: Which row is the header? (0-based, or -1 if none)
3. data_start_row: Which row does data start? (0-based)
4. column_structure: Array of columns with position, type, and description

Column types: item_code, description, unit, quantity, rate, amount

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
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            raw_text = response.text
            return _clean_json_response(raw_text)
        except Exception as e:
            return json.dumps({"error": f"Structure analysis failed: {str(e)}"})


# === TOOL 2: Extract Project Info ===
class ExtractProjectInfoInput(BaseModel):
    text: str = Field(description="Text sample from BOQ file")


class ExtractProjectInfoTool(BaseTool):
    name: str = "extract_project_info"
    description: str = (
        "Extracts project information from text. "
        "Returns JSON with project_name, project_code, client_name, start_year, end_year."
    )
    args_schema: Type[BaseModel] = ExtractProjectInfoInput

    def _run(self, text: str) -> str:
        prompt = f"""Extract project information from this BOQ text.

Text:
{text}

Return JSON with:
- project_name: string (or generate descriptive name)
- project_code: string (or generate like "PROJ-YYYY-XXX")
- client_name: string or null
- start_year: integer or null
- end_year: integer or null

Return ONLY valid JSON."""

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            raw_text = response.text
            return _clean_json_response(raw_text)
        except Exception as e:
            return json.dumps({"error": f"Project extraction failed: {str(e)}"})


# === TOOL 3: Extract Location Info ===
class ExtractLocationInfoInput(BaseModel):
    text: str = Field(description="Text sample from BOQ file")


class ExtractLocationInfoTool(BaseTool):
    name: str = "extract_location_info"
    description: str = (
        "Extracts location information from text. "
        "Returns JSON with location_name, city, state."
    )
    args_schema: Type[BaseModel] = ExtractLocationInfoInput

    def _run(self, text: str) -> str:
        prompt = f"""Extract location from this BOQ text.

Text:
{text}

Return JSON with:
- location_name: string (or "Unknown")
- city: string or null
- state: string or null

Return ONLY valid JSON."""

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            raw_text = response.text
            return _clean_json_response(raw_text)
        except Exception as e:
            return json.dumps({"error": f"Location extraction failed: {str(e)}"})