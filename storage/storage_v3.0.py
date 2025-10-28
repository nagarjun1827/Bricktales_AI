import os
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import google.generativeai as genai
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import time

load_dotenv()


@dataclass
class ProjectInfo:
    project_name: str
    project_code: str
    client_id: Optional[int] = None
    client_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    version: int = 1
    created_by: str = 'system'


@dataclass
class LocationInfo:
    project_id: int
    location_name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_by: str = 'system'


@dataclass
class BOQFileInfo:
    project_id: int
    file_name: str
    file_path: str
    file_type: str = 'xlsx'
    version: int = 1
    is_active: bool = True
    created_by: str = 'system'


@dataclass
class BOQItem:
    """Structured BOQ item information
    
    Note: supply_amount, labour_amount, and total_amount are GENERATED columns
    in the database and will be calculated automatically. We don't insert them.
    """
    boq_id: int
    item_code: Optional[str]
    item_description: str
    unit_of_measurement: str
    quantity: float
    supply_unit_rate: float = 0.0
    labour_unit_rate: float = 0.0
    location_id: Optional[int] = None
    created_by: str = 'system'
    
    # These are calculated locally for display purposes only
    # They are NOT inserted into the database (generated columns)
    supply_amount: float = 0.0
    labour_amount: float = 0.0
    total_amount: float = 0.0
    
    def calculate_amounts(self):
        """Calculate amounts for local display (not for database insertion)"""
        self.supply_amount = self.supply_unit_rate * self.quantity
        self.labour_amount = self.labour_unit_rate * self.quantity
        self.total_amount = self.supply_amount + self.labour_amount


class BaseAgent:
    def __init__(self, model_name: str = 'gemini-2.0-flash-exp'):
        self.model_name = model_name
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)


class SheetStructureAnalyzerAgent(BaseAgent):
    """Analyzes the entire sheet structure to understand data layout"""
    
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sheet to find headers and data regions"""
        
        print(f"      🔍 Analyzing sheet structure (shape: {df.shape})...")
        
        # Get first 30 rows for analysis
        sample_rows = min(30, len(df))
        sheet_text = []
        
        for idx in range(sample_rows):
            row_vals = [str(v) for v in df.iloc[idx].values if pd.notna(v)]
            if row_vals:
                sheet_text.append(f"Row {idx}: {' | '.join(row_vals[:10])}")  # First 10 values
        
        analysis_text = '\n'.join(sheet_text)
        
        prompt = f"""You are a BOQ (Bill of Quantities) sheet structure analyzer. Analyze this Excel sheet data.

Sheet Data:
{analysis_text}

Determine:
1. **has_header**: Does this sheet have a header row with column names? (true/false)
2. **header_row**: If has_header is true, which row number is the header? (0-based index, or -1 if no header)
3. **data_start_row**: Which row does the actual data start? (0-based index)
4. **column_structure**: What type of columns exist? Describe each column position.

For example:
- If Row 0 has "Item No, Description, Unit, Quantity, Rate, Amount" → has_header=true, header_row=0, data_start_row=1
- If Row 0 has "1, Supply wiring, m, 200, 45, 9000" → has_header=false, header_row=-1, data_start_row=0
- If Row 0-2 has title/project info and Row 3 has headers → header_row=3, data_start_row=4

Return a JSON object:
{{
    "has_header": true/false,
    "header_row": number (-1 if no header),
    "data_start_row": number,
    "column_structure": [
        {{"position": 0, "type": "item_code", "description": "Item numbers"}},
        {{"position": 1, "type": "description", "description": "Work description"}},
        {{"position": 2, "type": "unit", "description": "Unit of measurement"}},
        {{"position": 3, "type": "quantity", "description": "Quantity"}},
        {{"position": 4, "type": "rate", "description": "Rate per unit"}},
        {{"position": 5, "type": "amount", "description": "Total amount"}}
    ]
}}

Return ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean JSON
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            structure = json.loads(result_text)
            
            print(f"      ✓ Has header: {structure['has_header']}")
            print(f"      ✓ Header row: {structure['header_row']}")
            print(f"      ✓ Data starts at row: {structure['data_start_row']}")
            print(f"      ✓ Identified {len(structure['column_structure'])} columns")
            
            return structure
            
        except Exception as e:
            print(f"      ⚠ Structure analysis failed: {e}")
            # Fallback to simple heuristic
            return self._fallback_analysis(df)
    
    def _fallback_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback heuristic analysis"""
        
        # Check if first row looks like a header
        first_row = df.iloc[0]
        first_row_text = ' '.join([str(v).lower() for v in first_row.values if pd.notna(v)])
        
        header_keywords = ['item', 'description', 'quantity', 'unit', 'rate', 'amount', 'code', 'no']
        has_header_keyword = any(kw in first_row_text for kw in header_keywords)
        
        # Check if first row has mostly text (likely header) vs numbers (likely data)
        non_numeric_count = sum(1 for v in first_row.values 
                               if pd.notna(v) and not str(v).replace('.', '').replace('-', '').isdigit())
        
        has_header = has_header_keyword or (non_numeric_count > len(first_row) * 0.5)
        
        if has_header:
            return {
                'has_header': True,
                'header_row': 0,
                'data_start_row': 1,
                'column_structure': self._infer_columns_from_header(df.iloc[0])
            }
        else:
            return {
                'has_header': False,
                'header_row': -1,
                'data_start_row': 0,
                'column_structure': self._infer_columns_from_data(df)
            }
    
    def _infer_columns_from_header(self, header_row) -> List[Dict]:
        """Infer column structure from header row"""
        columns = []
        type_map = {
            'item': 'item_code',
            'code': 'item_code',
            'no': 'item_code',
            'description': 'description',
            'particular': 'description',
            'work': 'description',
            'unit': 'unit',
            'uom': 'unit',
            'quantity': 'quantity',
            'qty': 'quantity',
            'rate': 'rate',
            'price': 'rate',
            'amount': 'amount',
            'total': 'amount',
            'value': 'amount'
        }
        
        for idx, col_name in enumerate(header_row.values):
            if pd.isna(col_name):
                continue
            
            col_str = str(col_name).lower()
            col_type = 'unknown'
            
            for keyword, type_name in type_map.items():
                if keyword in col_str:
                    col_type = type_name
                    break
            
            columns.append({
                'position': idx,
                'type': col_type,
                'description': str(col_name)
            })
        
        return columns
    
    def _infer_columns_from_data(self, df: pd.DataFrame) -> List[Dict]:
        """Infer column structure from data patterns"""
        columns = []
        
        # Analyze first few rows to understand data types
        for idx in range(len(df.columns)):
            sample_values = df.iloc[:10, idx].dropna()
            
            if len(sample_values) == 0:
                continue
            
            # Check patterns
            is_numeric = all(str(v).replace('.', '').replace('-', '').isdigit() 
                           for v in sample_values)
            is_short_text = all(len(str(v)) < 20 for v in sample_values)
            is_long_text = any(len(str(v)) > 50 for v in sample_values)
            
            # Infer type
            if idx == 0 and is_numeric:
                col_type = 'item_code'
            elif is_long_text:
                col_type = 'description'
            elif is_short_text and not is_numeric:
                col_type = 'unit'
            elif is_numeric:
                # Could be quantity, rate, or amount
                # Typically: quantity < rate < amount in magnitude
                col_type = 'quantity' if idx <= 3 else 'rate' if idx == 4 else 'amount'
            else:
                col_type = 'unknown'
            
            columns.append({
                'position': idx,
                'type': col_type,
                'description': f'Column {idx}'
            })
        
        return columns


class SmartItemExtractorAgent(BaseAgent):
    """Extracts items intelligently based on sheet structure"""
    
    def execute(self, df: pd.DataFrame, structure: Dict[str, Any], 
                boq_id: int, location_id: int) -> List[BOQItem]:
        """Extract items using structure information"""
        
        print(f"      🔍 Extracting items with intelligent parsing...")
        
        # Get data rows
        data_start = structure['data_start_row']
        column_structure = structure['column_structure']
        
        # Create column mapping
        col_map = {}
        for col_info in column_structure:
            col_map[col_info['type']] = col_info['position']
        
        print(f"      ✓ Column mapping: {col_map}")
        
        items = []
        data_df = df.iloc[data_start:]
        
        for idx, row in data_df.iterrows():
            # Skip empty rows
            if row.isna().all():
                continue
            
            # Extract values based on column mapping
            item_code = self._get_value(row, col_map.get('item_code'))
            description = self._get_value(row, col_map.get('description'), default='')
            unit = self._get_value(row, col_map.get('unit'), default='Each')
            quantity = self._extract_numeric(self._get_value(row, col_map.get('quantity')))
            rate = self._extract_numeric(self._get_value(row, col_map.get('rate')))
            amount = self._extract_numeric(self._get_value(row, col_map.get('amount')))
            
            # Skip if no meaningful data
            # Need at least description and either (quantity or rate or amount)
            if not description or (quantity == 0 and rate == 0 and amount == 0):
                continue
            
            # Skip if description is too short (likely header or noise)
            if len(str(description).strip()) < 5:
                continue
            
            # Create item
            item = BOQItem(
                boq_id=boq_id,
                item_code=str(item_code) if item_code is not None else None,
                item_description=str(description).strip(),
                unit_of_measurement=self._normalize_unit(unit),
                quantity=quantity if quantity > 0 else (amount / rate if rate > 0 else 0),
                supply_unit_rate=rate,
                location_id=location_id,
                created_by='system'
            )
            
            item.calculate_amounts()
            items.append(item)
        
        print(f"      ✓ Extracted {len(items)} valid items")
        return items
    
    def _get_value(self, row, position, default=None):
        """Safely get value from row at position"""
        if position is None or position >= len(row):
            return default
        val = row.iloc[position]
        return val if pd.notna(val) else default
    
    def _extract_numeric(self, value) -> float:
        """Extract numeric value"""
        if value is None or pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).replace(',', '')
        match = re.search(r'[-+]?[0-9]*\.?[0-9]+', value_str)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return 0.0
        return 0.0
    
    def _normalize_unit(self, unit) -> str:
        """Normalize unit of measurement"""
        if not unit or pd.isna(unit):
            return "Each"
        
        unit_str = str(unit).strip()
        if unit_str.replace('.', '').replace('-', '').isdigit():
            return "Each"
        
        unit_lower = unit_str.lower()
        unit_map = {
            'sqm': 'Sqm', 'sq.m': 'Sqm', 'sq m': 'Sqm', 'square meter': 'Sqm',
            'cum': 'Cum', 'cu.m': 'Cum', 'cu m': 'Cum', 'cubic meter': 'Cum',
            'mtr': 'Mtr', 'm': 'Mtr', 'meter': 'Mtr', 'metre': 'Mtr',
            'kg': 'Kg', 'kilogram': 'Kg',
            'nos': 'Nos', 'no': 'Nos', 'number': 'Nos', 'each': 'Each',
            'litre': 'Ltr', 'ltr': 'Ltr', 'l': 'Ltr',
            'ton': 'Ton', 'tonne': 'Ton', 'mt': 'Ton',
            'rm': 'Rmt', 'rmt': 'Rmt', 'running meter': 'Rmt',
        }
        return unit_map.get(unit_lower, unit_str)


class ProjectExtractorAgent(BaseAgent):
    """Extracts project information"""
    
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
            result_text = response.text.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
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
            print(f"      ⚠ Project extraction failed: {e}")
            return ProjectInfo(
                project_name='BOQ Project',
                project_code=f'PROJ-{datetime.now().year}-{datetime.now().month:02d}{datetime.now().day:02d}'
            )


class LocationExtractorAgent(BaseAgent):
    """Extracts location information"""
    
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
            result_text = response.text.strip()
            
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            data = json.loads(result_text)
            
            address_parts = [data.get('location_name', 'Unknown')]
            if data.get('city'):
                address_parts.append(data['city'])
            if data.get('state'):
                address_parts.append(data['state'])
            
            return LocationInfo(
                project_id=project_id,
                location_name=data.get('location_name', 'Unknown'),
                address=', '.join(address_parts)
            )
        except Exception as e:
            print(f"      ⚠ Location extraction failed: {e}")
            return LocationInfo(
                project_id=project_id,
                location_name='Unknown',
                address='Unknown'
            )


class BoQDatabase:
    """Database operations"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'boq_database'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'root'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        self.conn = None
        self.cursor = None
    
    def connect(self):
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def insert_project(self, project_info: ProjectInfo) -> int:
        query = """
            INSERT INTO projects (
                project_name, project_code, client_id, client_name,
                start_date, end_date, version, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING project_id
        """
        
        self.cursor.execute(query, (
            project_info.project_name,
            project_info.project_code,
            project_info.client_id,
            project_info.client_name,
            project_info.start_date,
            project_info.end_date,
            project_info.version,
            project_info.created_by
        ))
        
        project_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return project_id
    
    def insert_location(self, location_info: LocationInfo) -> int:
        query = """
            INSERT INTO locations (
                project_id, location_name, address, latitude, longitude, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING location_id
        """
        
        self.cursor.execute(query, (
            location_info.project_id,
            location_info.location_name,
            location_info.address,
            location_info.latitude,
            location_info.longitude,
            location_info.created_by
        ))
        
        location_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return location_id
    
    def insert_boq_file(self, file_info: BOQFileInfo) -> int:
        query = """
            INSERT INTO store_boq_files (
                project_id, file_name, file_path, file_type,
                version, is_active, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING boq_id
        """
        
        self.cursor.execute(query, (
            file_info.project_id,
            file_info.file_name,
            file_info.file_path,
            file_info.file_type,
            file_info.version,
            file_info.is_active,
            file_info.created_by
        ))
        
        boq_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return boq_id
    
    def insert_boq_items_batch(self, items: List[BOQItem]):
        if not items:
            return
        
        # Don't include generated columns: supply_amount, labour_amount, total_amount
        # These are calculated automatically by the database
        items_data = [
            (
                item.boq_id, 
                item.item_code, 
                item.item_description,
                item.unit_of_measurement, 
                item.quantity,
                item.supply_unit_rate,
                item.labour_unit_rate,
                item.location_id, 
                item.created_by
            )
            for item in items
        ]
        
        query = """
            INSERT INTO store_boq_items (
                boq_id, item_code, item_description, unit_of_measurement,
                quantity, supply_unit_rate, labour_unit_rate,
                location_id, created_by
            ) VALUES %s
        """
        
        execute_values(self.cursor, query, items_data)
        self.conn.commit()
    
    def get_boq_totals(self, boq_id: int) -> Dict[str, float]:
        """Get calculated totals for a BOQ file from the database"""
        query = """
            SELECT 
                COUNT(*) as item_count,
                COALESCE(SUM(supply_amount), 0) as total_supply,
                COALESCE(SUM(labour_amount), 0) as total_labour,
                COALESCE(SUM(total_amount), 0) as total_amount
            FROM store_boq_items
            WHERE boq_id = %s
        """
        
        self.cursor.execute(query, (boq_id,))
        result = self.cursor.fetchone()
        
        return {
            'item_count': result[0],
            'total_supply': float(result[1]),
            'total_labour': float(result[2]),
            'total_amount': float(result[3])
        }


class IntelligentBOQProcessor:
    """Main processor with intelligent agents"""
    
    def __init__(self):
        self.structure_agent = SheetStructureAnalyzerAgent()
        self.item_agent = SmartItemExtractorAgent()
        self.project_agent = ProjectExtractorAgent()
        self.location_agent = LocationExtractorAgent()
        self.db = BoQDatabase()
    
    def process_file(self, file_path: str, uploaded_by: str = 'system') -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"🤖 INTELLIGENT MULTI-AGENT BOQ PROCESSOR")
        print(f"{'='*70}")
        print(f"File: {Path(file_path).name}\n")
        
        start_time = time.time()
        
        try:
            self.db.connect()
            
            # Step 1: Read file
            print("🔍 AGENT: File Reader")
            excel_file = pd.ExcelFile(file_path)
            sheets = {name: pd.read_excel(file_path, sheet_name=name) 
                     for name in excel_file.sheet_names}
            print(f"   ✓ Read {len(sheets)} sheets: {', '.join(sheets.keys())}\n")
            
            # Step 2: Extract project info
            print("🔍 AGENT: Project Extractor")
            first_sheet = list(sheets.values())[0]
            text_sample = self._extract_text(first_sheet, 50)
            project_info = self.project_agent.execute(text_sample)
            print(f"   ✓ Project: {project_info.project_name}")
            print(f"   ✓ Code: {project_info.project_code}\n")
            
            # Step 3: Insert project
            print("💾 DATABASE: Inserting project")
            project_id = self.db.insert_project(project_info)
            print(f"   ✓ Project ID: {project_id}\n")
            
            # Step 4: Extract location
            print("🔍 AGENT: Location Extractor")
            location_info = self.location_agent.execute(text_sample, project_id)
            print(f"   ✓ Location: {location_info.location_name}\n")
            
            # Step 5: Insert location
            print("💾 DATABASE: Inserting location")
            location_id = self.db.insert_location(location_info)
            print(f"   ✓ Location ID: {location_id}\n")
            
            # Step 6: Insert file
            print("💾 DATABASE: Inserting file metadata")
            file_info = BOQFileInfo(
                project_id=project_id,
                file_name=Path(file_path).name,
                file_path=file_path,
                created_by=uploaded_by
            )
            boq_id = self.db.insert_boq_file(file_info)
            print(f"   ✓ BOQ File ID: {boq_id}\n")
            
            # Step 7: Filter and process sheets
            print("🔍 AGENT: Sheet Filter")
            boq_sheets = self._filter_sheets(sheets)
            print(f"   ✓ Processing {len(boq_sheets)} BOQ sheet(s)\n")
            
            # Step 8: Process each sheet
            all_items = []
            for sheet_name, sheet_df in boq_sheets.items():
                print(f"   📄 Processing: {sheet_name}")
                
                # Analyze structure
                structure = self.structure_agent.execute(sheet_df)
                
                # Extract items
                items = self.item_agent.execute(sheet_df, structure, boq_id, location_id)
                all_items.extend(items)
                print()
            
            # Step 9: Insert items
            print("💾 DATABASE: Inserting BOQ items")
            self.db.insert_boq_items_batch(all_items)
            print(f"   ✓ Inserted {len(all_items)} items\n")
            
            # Step 10: Get calculated totals from database
            print("📊 Fetching calculated totals from database...")
            totals = self.db.get_boq_totals(boq_id)
            
            # Summary
            elapsed = time.time() - start_time
            
            print(f"{'='*70}")
            print(f"✓ PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"Project ID:      {project_id}")
            print(f"Location ID:     {location_id}")
            print(f"BOQ File ID:     {boq_id}")
            print(f"Total Items:     {totals['item_count']}")
            print(f"Supply Amount:   ₹{totals['total_supply']:,.2f}")
            print(f"Labour Amount:   ₹{totals['total_labour']:,.2f}")
            print(f"Total Amount:    ₹{totals['total_amount']:,.2f}")
            print(f"Time:            {elapsed:.2f}s")
            print(f"{'='*70}\n")
            
            return {
                'success': True,
                'project_id': project_id,
                'boq_id': boq_id,
                'total_items': totals['item_count'],
                'total_supply': totals['total_supply'],
                'total_labour': totals['total_labour'],
                'total_amount': totals['total_amount'],
                'processing_time': elapsed
            }
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        
        finally:
            self.db.close()
    
    def _extract_text(self, df: pd.DataFrame, max_rows: int) -> str:
        lines = []
        for idx, row in df.head(max_rows).iterrows():
            text = ' '.join([str(v) for v in row.values if pd.notna(v)])
            if text.strip():
                lines.append(text)
        return '\n'.join(lines)[:5000]
    
    def _filter_sheets(self, sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        skip = ['summary', 'assumption', 'note', 'index', 'cover', 'terms']
        boq_sheets = {}
        for name, df in sheets.items():
            if any(kw in name.lower() for kw in skip):
                continue
            if len(df) < 5 or len(df.columns) < 3:
                continue
            boq_sheets[name] = df
        return boq_sheets


def main():
    import sys
    
    if not os.getenv('GEMINI_API_KEY'):
        print("✗ Error: GEMINI_API_KEY not set")
        return
    
    file_path = sys.argv[1] if len(sys.argv) > 1 else './storage_input_files/sample.xlsx'
    
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        return
    
    processor = IntelligentBOQProcessor()
    result = processor.process_file(file_path)
    
    if not result['success']:
        print(f"\n✗ Failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()