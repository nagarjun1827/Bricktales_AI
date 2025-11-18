# utils/llm_utils.py
import re
import json
import datetime
from typing import Dict
from config import qa_agent_model

def get_json_from_text(text: str) -> Dict:
    """Uses Gemini to extract structured JSON from raw text"""
    prompt_text = text[:12000]

    prompt = f"""
    You are an expert data extractor. Analyze the following tender document text 
    and extract the specified fields.
    
    Respond ONLY with a single, valid JSON object.
    
    Fields to extract:
    - "tender_date": The main date of the tender notice (YYYY-MM-DD). If not found, use null.
    - "submission_deadline": The final date and time for submission (ISO 8601 format: YYYY-MM-DDTHH:MM:SS). If not found, use null.
    - "tender_status": The current status. Default to "Open".
    - "tender_value": The estimated value of the tender as a decimal number (e.g., 125000.00). If not found, use 0.00.
    - "file_name": A good, short name for this document, e.g., "Main Tender Document" or "Addendum 1".
    
    Text:
    "{prompt_text}"
    
    JSON:
    """
    
    try:
        response = qa_agent_model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            return {}
            
        json_data = json.loads(json_match.group(0))
        
        if 'tender_date' in json_data and json_data['tender_date']:
            try:
                json_data['tender_date'] = datetime.datetime.fromisoformat(
                    str(json_data['tender_date'])
                ).date()
            except:
                json_data['tender_date'] = None
        
        if 'submission_deadline' in json_data and json_data['submission_deadline']:
            try:
                deadline_str = str(json_data['submission_deadline']).replace('Z', '+00:00')
                if '+' not in deadline_str:
                    deadline_str += '+00:00'
                json_data['submission_deadline'] = datetime.datetime.fromisoformat(deadline_str)
            except:
                json_data['submission_deadline'] = None
        
        if 'tender_value' in json_data and json_data['tender_value']:
            try:
                value_str = str(json_data['tender_value'])
                value_str = re.sub(r'[^\d.]', '', value_str)
                json_data['tender_value'] = float(value_str)
            except:
                json_data['tender_value'] = 0.00
        
        return json_data
        
    except Exception as e:
        print(f"   ⚠️ Error extracting JSON: {e}")
        return {}