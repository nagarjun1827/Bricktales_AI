from typing import Any
from models.domain import LocationInfo
from .base_agent import BaseAgent
import json

class LocationExtractorAgent(BaseAgent):
    def execute(self, text: str, project_id: int) -> LocationInfo:
        prompt = f"""Extract location from this text.

Text:
{text}

Return JSON with:
- location_name: string (or "Unknown" if not found)
- city: string or null
- state: string or null

Return ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            result_text = self._clean_json_response(response.text)
            data = json.loads(result_text)
            
            address_parts = [part for part in [data.get('location_name'), data.get('city'), data.get('state')] if part]
            address = ', '.join(address_parts) if address_parts else 'Unknown'
            
            return LocationInfo(
                project_id=project_id,
                location_name=data.get('location_name', 'Unknown'),
                address=address
            )
        except Exception as e:
            print(f"      Location extraction failed: {e}")
            return LocationInfo(
                project_id=project_id,
                location_name='Unknown',
                address='Unknown'
            )