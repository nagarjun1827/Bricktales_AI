import time
import re
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import google.generativeai as genai

from app.core.config import settings
from app.models.tbe_domain import TBEProjectInfo, TBELocationInfo, TBEBOQFileInfo, TBEBOQItem
from app.repositories.tbe_repository import TBERepository
from app.services.tbe_pattern_matcher import TBEPatternMatcher


class TBEGeminiAgent:
    """Gemini agent for TBE BOQ extraction"""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def extract_project_info(self, text: str) -> Dict:
        """Extract project information using Gemini"""
        prompt = f"""Extract project information from this text.

Text:
{text}

Return JSON with:
- project_name: string
- project_code: string (or generate one)
- year: string
- location: string
- client_name: string or null

Return ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            result_text = self._clean_json(response.text)
            return json.loads(result_text)
        except Exception as e:
            print(f"  Gemini extraction failed: {e}")
            return {}
    
    def identify_columns(self, columns: List[str]) -> Dict:
        """Identify column mappings using Gemini"""
        columns_str = ', '.join([f'"{col}"' for col in columns])
        
        prompt = f"""Map these columns to BOQ fields:
Columns: {columns_str}

Map to: item_code, description, quantity, unit

Return JSON mapping field to column name. Use null if not found.

Return ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            result_text = self._clean_json(response.text)
            return json.loads(result_text)
        except Exception as e:
            print(f"  Column identification failed: {e}")
            return {}
    
    def _clean_json(self, text: str) -> str:
        """Clean JSON response from Gemini"""
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()


class TBEService:
    """Service for processing To-Be-Estimated BOQ files"""
    
    def __init__(self):
        self.repo = TBERepository()
        self.agent = TBEGeminiAgent()
        self.pattern_matcher = TBEPatternMatcher()
    
    def process_file(self, file_path: str, uploaded_by: str = "system") -> Dict[str, Any]:
        """Process TBE BOQ file and store in database"""
        
        print(f"\n{'='*70}")
        print(f"TO-BE-ESTIMATED BOQ PROCESSOR")
        print(f"{'='*70}")
        print(f"File: {Path(file_path).name}")
        print(f"Mode: Quantity Only (No Pricing)")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        try:
            # Step 1: Read Excel file
            print("📖 Step 1: Reading Excel file...")
            excel_file = pd.ExcelFile(file_path)
            sheets = {
                name: pd.read_excel(file_path, sheet_name=name) 
                for name in excel_file.sheet_names
            }
            print(f"   ✓ Read {len(sheets)} sheets: {', '.join(sheets.keys())}\n")
            
            # Step 2: Extract project info
            print("🔍 Step 2: Extracting project information...")
            first_sheet = list(sheets.values())[0]
            project_info = self._extract_project_info(first_sheet)
            print(f"   ✓ Project: {project_info.project_name}")
            print(f"   ✓ Code: {project_info.project_code}")
            print(f"   ✓ Location: {project_info.client_name or 'N/A'}\n")
            
            # Step 3: Insert project
            print("💾 Step 3: Inserting project...")
            project_id = self.repo.insert_project(project_info)
            print(f"   ✓ Project ID: {project_id}\n")
            
            # Step 4: Extract and insert location
            print("💾 Step 4: Inserting location...")
            location_info = self._extract_location_info(first_sheet, project_id)
            location_id = self.repo.insert_location(location_info)
            print(f"   ✓ Location ID: {location_id}\n")
            
            # Step 5: Insert file record
            print("💾 Step 5: Creating TBE BOQ file record...")
            file_info = TBEBOQFileInfo(
                project_id=project_id,
                file_name=Path(file_path).name,
                file_path=file_path,
                created_by=uploaded_by
            )
            boq_id = self.repo.insert_tbe_boq_file(file_info)
            print(f"   ✓ BOQ ID: {boq_id}\n")
            
            # Step 6: Extract items from all sheets
            print("🔍 Step 6: Extracting BOQ items (quantities only)...")
            all_items = []
            
            for sheet_name, sheet_df in sheets.items():
                if self._should_skip_sheet(sheet_name, sheet_df):
                    print(f"   ⊗ Skipping: {sheet_name}")
                    continue
                
                print(f"\n   📄 Processing: {sheet_name}")
                items = self._extract_items_from_sheet(sheet_df, boq_id, location_id)
                all_items.extend(items)
            
            # Step 7: Insert items
            print(f"\n💾 Step 7: Inserting {len(all_items)} items...")
            self.repo.insert_tbe_items_batch(all_items)
            print(f"   ✓ Inserted {len(all_items)} items\n")
            
            # Step 8: Get summary
            print("📊 Step 8: Fetching summary...")
            summary = self.repo.get_tbe_boq_summary(boq_id)
            
            elapsed_total = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"✓ PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"Project ID:      {project_id}")
            print(f"Location ID:     {location_id}")
            print(f"BOQ ID:          {boq_id}")
            print(f"Total Items:     {summary['item_count']}")
            print(f"Processing Time: {elapsed_total:.2f}s")
            print(f"{'='*70}\n")
            
            return {
                'success': True,
                'project_id': project_id,
                'location_id': location_id,
                'boq_id': boq_id,
                'total_items': summary['item_count'],
                'processing_time': elapsed_total,
                'message': 'TBE BOQ processed successfully. Ready for rate estimation.'
            }
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_project_info(self, df: pd.DataFrame) -> TBEProjectInfo:
        """Extract project information from sheet"""
        project_info = {
            'project_name': None,
            'project_code': None,
            'year': None,
            'location': None,
            'client_name': None
        }
        
        # Try pattern matching first
        for idx, row in df.head(20).iterrows():
            for col in df.columns:
                cell_value = str(row[col]).strip() if pd.notna(row[col]) else ""
                cell_lower = cell_value.lower()
                
                if not project_info['project_name']:
                    if any(kw in cell_lower for kw in ['project', 'work', 'contract']):
                        if 10 < len(cell_value) < 200:
                            project_info['project_name'] = cell_value
                
                if not project_info['location']:
                    if any(kw in cell_lower for kw in ['location', 'site', 'place']):
                        next_col_idx = df.columns.get_loc(col) + 1
                        if next_col_idx < len(df.columns):
                            next_val = str(row[df.columns[next_col_idx]])
                            if pd.notna(row[df.columns[next_col_idx]]):
                                project_info['location'] = next_val
                
                if not project_info['year']:
                    year_match = re.search(r'20\d{2}(?:-20?\d{2})?', cell_value)
                    if year_match:
                        project_info['year'] = year_match.group()
        
        # Use Gemini if needed
        if not project_info['project_name'] or not project_info['location']:
            text = df.head(20).to_string()
            gemini_result = self.agent.extract_project_info(text)
            
            if gemini_result:
                for key in project_info.keys():
                    if not project_info[key] and gemini_result.get(key):
                        project_info[key] = gemini_result[key]
        
        # Set defaults
        if not project_info['project_name']:
            project_info['project_name'] = f"TBE Project {datetime.now().strftime('%Y%m%d')}"
        
        if not project_info['project_code']:
            year = project_info.get('year', datetime.now().year)
            project_info['project_code'] = f"TBE-{year}-{datetime.now().strftime('%m%d%H%M')}"
        
        if not project_info['location']:
            project_info['location'] = "Unknown Location"
        
        # Parse dates from year
        start_date = None
        end_date = None
        if project_info.get('year'):
            year_str = str(project_info['year'])
            if '-' in year_str:
                years = year_str.split('-')
                try:
                    start_date = f"{int(years[0])}-01-01"
                    end_date = f"{int(years[1])}-12-31"
                except ValueError:
                    start_date = f"{datetime.now().year}-01-01"
            else:
                try:
                    year = int(year_str)
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"
                except ValueError:
                    pass
        
        return TBEProjectInfo(
            project_name=project_info['project_name'],
            project_code=project_info['project_code'],
            client_name=project_info.get('client_name'),
            start_date=start_date,
            end_date=end_date
        )
    
    def _extract_location_info(self, df: pd.DataFrame, project_id: int) -> TBELocationInfo:
        """Extract location information"""
        location_name = "Unknown Location"
        
        for idx, row in df.head(20).iterrows():
            for col in df.columns:
                cell_value = str(row[col]).strip() if pd.notna(row[col]) else ""
                cell_lower = cell_value.lower()
                
                if any(kw in cell_lower for kw in ['location', 'site', 'place']):
                    next_col_idx = df.columns.get_loc(col) + 1
                    if next_col_idx < len(df.columns):
                        next_val = str(row[df.columns[next_col_idx]])
                        if pd.notna(row[df.columns[next_col_idx]]):
                            location_name = next_val
                            break
        
        return TBELocationInfo(
            project_id=project_id,
            location_name=location_name,
            address=location_name
        )
    
    def _should_skip_sheet(self, sheet_name: str, sheet_df: pd.DataFrame) -> bool:
        """Check if sheet should be skipped"""
        skip_keywords = ['summary', 'assumption', 'note', 'index', 'cover']
        
        if any(kw in sheet_name.lower() for kw in skip_keywords):
            return True
        
        if len(sheet_df) < 5 or len(sheet_df.columns) < 3:
            return True
        
        return False
    
    def _extract_items_from_sheet(
        self, 
        df: pd.DataFrame, 
        boq_id: int, 
        location_id: int
    ) -> List[TBEBOQItem]:
        """Extract items from a sheet"""
        print(f"  Analyzing sheet: {len(df)} rows, {len(df.columns)} columns")
        
        # Identify columns
        column_mapping = self.pattern_matcher.identify_columns(df)
        print(f"  Column mapping: {column_mapping}")
        
        # Use Gemini if pattern matching didn't find key columns
        if not column_mapping['description'] or not column_mapping['quantity']:
            print(f"  Using Gemini for column identification...")
            gemini_mapping = self.agent.identify_columns(df.columns.tolist())
            
            if gemini_mapping:
                for key, value in gemini_mapping.items():
                    if value and not column_mapping.get(key):
                        column_mapping[key] = value
                print(f"  Updated mapping: {column_mapping}")
        
        # Extract items
        items = []
        for idx, row in df.iterrows():
            if column_mapping['description'] and pd.isna(row[column_mapping['description']]):
                continue
            
            # Check if it's a valid item row
            is_item = False
            
            if column_mapping['item_code']:
                if self.pattern_matcher.is_valid_item_code(row[column_mapping['item_code']]):
                    is_item = True
            
            if column_mapping['quantity']:
                qty = self.pattern_matcher.extract_numeric(row[column_mapping['quantity']])
                if qty > 0:
                    is_item = True
            
            if not is_item:
                continue
            
            # Extract values
            item_code_val = row[column_mapping['item_code']] if column_mapping['item_code'] and pd.notna(row[column_mapping['item_code']]) else None
            item_code = str(item_code_val) if item_code_val is not None else None
            
            desc_val = row[column_mapping['description']] if column_mapping['description'] else ""
            description = str(desc_val) if pd.notna(desc_val) else ""
            
            quantity = self.pattern_matcher.extract_numeric(
                row[column_mapping['quantity']]
            ) if column_mapping['quantity'] else 0
            
            unit = self.pattern_matcher.normalize_unit(
                row[column_mapping['unit']]
            ) if column_mapping['unit'] else "Each"
            
            if quantity == 0:
                continue
            
            # Clean description
            description = re.sub(r'\s+', ' ', description).strip()
            
            item = TBEBOQItem(
                boq_id=boq_id,
                item_code=item_code,
                item_description=description,
                unit_of_measurement=unit,
                quantity=quantity,
                location_id=location_id
            )
            
            items.append(item)
        
        print(f"  Extracted {len(items)} items")
        return items
    
    def get_tbe_items(self, boq_id: int, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get TBE items with pagination"""
        try:
            items = self.repo.get_tbe_items_by_boq(boq_id, limit, offset)
            summary = self.repo.get_tbe_boq_summary(boq_id)
            
            return {
                "success": True,
                "boq_id": boq_id,
                "total_items": summary["item_count"],
                "returned_items": len(items),
                "items": items,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + len(items)) < summary["item_count"]
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }