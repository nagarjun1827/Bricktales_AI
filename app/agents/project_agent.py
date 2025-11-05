from typing import Any
from models.domain import ProjectInfo
from .base_agent import BaseAgent
from datetime import datetime
import json

class ProjectExtractorAgent(BaseAgent):
    def execute(self, text: str) -> ProjectInfo:
        prompt = f"""Extract project information from this text.

Text:
{text}

Return JSON with:
- project_name: string (or generate a descriptive name if not found)
- project_code: string (or generate like "PROJ-YYYY-XXX")
- client_name: string or null
- start_year: integer or null
- end_year: integer or null

Return ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            result_text = self._clean_json_response(response.text)
            data = json.loads(result_text)
            
            start_date = None
            end_date = None
            if data.get('start_year'):
                start_date = f"{data['start_year']}-01-01"
            if data.get('end_year'):
                end_date = f"{data['end_year']}-12-31"
            elif data.get('start_year'):
                end_date = f"{data['start_year']}-12-31"
            
            return ProjectInfo(
                project_name=data.get('project_name', 'BOQ Project'),
                project_code=data.get('project_code', f'PROJ-{datetime.now().year}-{datetime.now().month:02d}{datetime.now().day:02d}'),
                client_name=data.get('client_name'),
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            print(f"      Project extraction failed: {e}")
            return ProjectInfo(
                project_name='BOQ Project',
                project_code=f'PROJ-{datetime.now().year}-{datetime.now().month:02d}{datetime.now().day:02d}'
            )