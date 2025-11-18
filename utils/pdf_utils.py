# utils/pdf_utils.py
import re
import requests
import PyPDF2
from io import BytesIO
from typing import Optional

def fetch_pdf_text(url: str) -> Optional[str]:
    """Download PDF from URL and extract text"""
    try:
        print(f"Fetching PDF from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=30, headers=headers, verify=False)
        response.raise_for_status()
        
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        raw_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                raw_text += f"\n--- Page {page_num + 1} ---\n"
                raw_text += page_text
        
        raw_text = re.sub(r'(\n\s*)+\n', '\n', raw_text)
        print(f"Extracted {len(raw_text)} characters")
        return raw_text
        
    except Exception as e:
        print(f"❌ Error fetching PDF: {e}")
        return None