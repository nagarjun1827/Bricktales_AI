"""
Service for processing To-Be-Estimated BOQ files (quantities only, no rates).
"""
import time
import re
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import google.generativeai as genai

from app.core.settings import settings
from app.models.tbe_domain import TBEProjectInfo, TBELocationInfo, TBEBOQFileInfo, TBEBOQItem
from app.repositories.tbe_boq_repository import TBEBOQRepository
from app.services.pattern_matcher import PatternMatcher


class GeminiExtractor:
    """Gemini AI extractor for TBE BOQ."""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def extract_project_info(self, text: str) -> Dict:
        """Extract project information."""
        prompt = f"""Extract project info from: {text}
Return JSON: {{"project_name": str, "project_code": str, "year": str, "location": str, "client_name": str|null}}"""

        try:
            response = self.model.generate_content(prompt)
            return json.loads(self._clean_json(response.text))
        except:
            return {}
    
    def identify_columns(self, columns: List[str]) -> Dict:
        """Identify column mappings."""
        cols = ', '.join([f'"{c}"' for c in columns])
        prompt = f"""Map columns to: item_code, description, quantity, unit
Columns: {cols}
Return JSON mapping."""

        try:
            response = self.model.generate_content(prompt)
            return json.loads(self._clean_json(response.text))
        except:
            return {}
    
    @staticmethod
    def _clean_json(text: str) -> str:
        """Clean JSON response."""
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()


class TBEBOQProcessor:
    """Processes To-Be-Estimated BOQ files."""
    
    def __init__(self):
        self.repo = TBEBOQRepository()
        self.extractor = GeminiExtractor()
        self.pattern_matcher = PatternMatcher()
    
    def process_file(self, file_path: str, uploaded_by: str = "system") -> Dict[str, Any]:
        """Process TBE BOQ file."""
        print(f"\n{'='*70}")
        print(f"TO-BE-ESTIMATED BOQ PROCESSOR")
        print(f"{'='*70}")
        print(f"File: {Path(file_path).name}\n")
        
        start_time = time.time()
        
        try:
            # Read file
            print("📖 Reading Excel file...")
            sheets = self._read_excel(file_path)
            print(f"   ✓ Read {len(sheets)} sheets\n")
            
            # Extract metadata
            first_sheet = list(sheets.values())[0]
            project_id, location_id = self._process_metadata(first_sheet, uploaded_by)
            
            # Create BOQ file
            print("💾 Creating TBE BOQ file record...")
            boq_id = self._create_boq_file(project_id, file_path, uploaded_by)
            print(f"   ✓ BOQ ID: {boq_id}\n")
            
            # Extract items
            print("📊 Extracting items...")
            all_items = []
            for name, df in sheets.items():
                if self._should_skip_sheet(name, df):
                    continue
                print(f"   📄 {name}")
                items = self._extract_items(df, boq_id, location_id)
                all_items.extend(items)
            
            # Insert items
            print(f"\n💾 Inserting {len(all_items)} items...")
            self.repo.insert_tbe_items_batch(all_items)
            
            # Summary
            summary = self.repo.get_tbe_boq_summary(boq_id)
            elapsed = time.time() - start_time
            
            self._print_summary(project_id, location_id, boq_id, summary, elapsed)
            
            return {
                'success': True,
                'project_id': project_id,
                'location_id': location_id,
                'boq_id': boq_id,
                'total_items': summary['item_count'],
                'processing_time': elapsed,
            }
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _read_excel(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read Excel file."""
        excel_file = pd.ExcelFile(file_path)
        return {
            name: pd.read_excel(file_path, sheet_name=name)
            for name in excel_file.sheet_names
        }
    
    def _process_metadata(self, df: pd.DataFrame, uploaded_by: str) -> tuple:
        """Extract and save project/location."""
        text = df.head(20).to_string()
        
        # Extract project
        project_data = self.extractor.extract_project_info(text)
        project_info = self._build_project_info(project_data)
        project_id = self.repo.insert_project(project_info)
        
        # Extract location
        location_info = self._build_location_info(project_data, project_id)
        location_id = self.repo.insert_location(location_info)
        
        return project_id, location_id
    
    def _create_boq_file(self, project_id: int, file_path: str, uploaded_by: str) -> int:
        """Create BOQ file record."""
        file_info = TBEBOQFileInfo(
            project_id=project_id,
            file_name=Path(file_path).name,
            file_path=file_path,
            created_by=uploaded_by
        )
        return self.repo.insert_tbe_boq_file(file_info)
    
    def _extract_items(self, df: pd.DataFrame, boq_id: int, location_id: int) -> List[TBEBOQItem]:
        """Extract items from sheet."""
        # Identify columns
        col_mapping = self.pattern_matcher.identify_columns(df)
        
        if not col_mapping['description']:
            gemini_mapping = self.extractor.identify_columns(df.columns.tolist())
            for key, value in gemini_mapping.items():
                if value and not col_mapping.get(key):
                    col_mapping[key] = value
        
        # Extract items
        items = []
        for _, row in df.iterrows():
            if col_mapping['description'] and pd.isna(row[col_mapping['description']]):
                continue
            
            # Get values
            item_code = str(row[col_mapping['item_code']]) if col_mapping['item_code'] and pd.notna(row[col_mapping['item_code']]) else None
            desc = str(row[col_mapping['description']]) if col_mapping['description'] and pd.notna(row[col_mapping['description']]) else ""
            quantity = self.pattern_matcher.extract_numeric(row[col_mapping['quantity']]) if col_mapping['quantity'] else 0
            unit = self.pattern_matcher.normalize_unit(row[col_mapping['unit']]) if col_mapping['unit'] else "Each"
            
            if quantity == 0 or not desc:
                continue
            
            items.append(TBEBOQItem(
                boq_id=boq_id,
                item_code=item_code,
                item_description=re.sub(r'\s+', ' ', desc).strip(),
                unit_of_measurement=unit,
                quantity=quantity,
                location_id=location_id
            ))
        
        print(f"      ✓ Extracted {len(items)} items")
        return items
    
    @staticmethod
    def _should_skip_sheet(name: str, df: pd.DataFrame) -> bool:
        """Check if sheet should be skipped."""
        skip = ['summary', 'assumption', 'note', 'index', 'cover']
        return any(kw in name.lower() for kw in skip) or len(df) < 5
    
    @staticmethod
    def _build_project_info(data: dict) -> TBEProjectInfo:
        """Build project info."""
        year = data.get('year', datetime.now().year)
        return TBEProjectInfo(
            project_name=data.get('project_name', f"TBE Project {datetime.now():%Y%m%d}"),
            project_code=data.get('project_code', f"TBE-{year}-{datetime.now():%m%d%H%M}"),
            client_name=data.get('client_name'),
            start_date=f"{year}-01-01" if year else None,
            end_date=f"{year}-12-31" if year else None,
        )
    
    @staticmethod
    def _build_location_info(data: dict, project_id: int) -> TBELocationInfo:
        """Build location info."""
        location = data.get('location', 'Unknown Location')
        return TBELocationInfo(
            project_id=project_id,
            location_name=location,
            address=location
        )
    
    @staticmethod
    def _print_summary(project_id, location_id, boq_id, summary, elapsed):
        """Print summary."""
        print(f"\n{'='*70}")
        print("✓ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Project ID:      {project_id}")
        print(f"Location ID:     {location_id}")
        print(f"BOQ ID:          {boq_id}")
        print(f"Total Items:     {summary['item_count']}")
        print(f"Processing Time: {elapsed:.2f}s")
        print(f"{'='*70}\n")