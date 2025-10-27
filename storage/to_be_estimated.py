import os
import re
import json
import pandas as pd
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

load_dotenv()


class BoQPatternMatcher:
    """Pattern matching for BoQ data extraction - no pricing"""
    
    COLUMN_PATTERNS = {
        'item_code': [
            r'item[\s_-]*no',
            r'item[\s_-]*code',
            r's[\s_-]*no',
            r'sl[\s_-]*no',
            r'serial[\s_-]*no',
            r'code',
            r'no\.',
        ],
        'description': [
            r'description',
            r'item[\s_-]*description',
            r'work[\s_-]*description',
            r'particulars',
            r'scope[\s_-]*of[\s_-]*work',
            r'details',
        ],
        'quantity': [
            r'qty',
            r'quantity',
            r'qnty',
            r'quan\.',
        ],
        'unit': [
            r'unit',
            r'uom',
            r'unit[\s_-]*of[\s_-]*measurement',
            r'measure',
        ]
    }
    
    ITEM_CODE_PATTERN = re.compile(r'^[A-Z]?[0-9]+(\.[0-9]+)*$')
    
    @staticmethod
    def normalize_column_name(col_name: str) -> str:
        if pd.isna(col_name):
            return ""
        return str(col_name).lower().strip()
    
    @classmethod
    def match_column(cls, col_name: str, field_type: str) -> bool:
        normalized = cls.normalize_column_name(col_name)
        patterns = cls.COLUMN_PATTERNS.get(field_type, [])
        
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def identify_columns(cls, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        columns = {
            'item_code': None,
            'description': None,
            'quantity': None,
            'unit': None
        }
        
        for col in df.columns:
            col_str = str(col)
            for field_type in columns.keys():
                if columns[field_type] is None and cls.match_column(col_str, field_type):
                    columns[field_type] = col
                    break
        
        return columns
    
    @classmethod
    def is_valid_item_code(cls, value) -> bool:
        if pd.isna(value):
            return False
        value_str = str(value).strip()
        return bool(cls.ITEM_CODE_PATTERN.match(value_str))
    
    @classmethod
    def normalize_unit(cls, unit: str) -> str:
        if pd.isna(unit):
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
    
    @classmethod
    def extract_numeric(cls, value) -> float:
        if pd.isna(value):
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


class BoQDatabase:
    """Database operations for to-be-estimated BoQ"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        if db_config is None:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'boq_database'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'root'),
                'port': int(os.getenv('DB_PORT', 5432))
            }
        else:
            self.db_config = db_config
        
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
    
    def insert_project(self, project_info: Dict) -> int:
        start_date = None
        end_date = None
        
        if project_info.get('year'):
            year_str = str(project_info['year'])
            
            if '-' in year_str and len(year_str.split('-')) == 2:
                years = year_str.split('-')
                try:
                    start_year = int(years[0])
                    end_year = int(years[1])
                    start_date = f"{start_year}-01-01"
                    end_date = f"{end_year}-12-31"
                except ValueError:
                    start_date = f"{datetime.now().year}-01-01"
            else:
                try:
                    year = int(year_str)
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"
                except ValueError:
                    start_date = f"{datetime.now().year}-01-01"
        
        query = """
        INSERT INTO projects (
            project_name, project_code, client_name, 
            start_date, end_date, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING project_id
        """
        
        self.cursor.execute(query, (
            project_info['project_name'],
            project_info['project_code'],
            project_info.get('client_name'),
            start_date,
            end_date,
            datetime.now()
        ))
        
        project_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return project_id
    
    def insert_location(self, project_id: int, location_info: Dict) -> int:
        query = """
        INSERT INTO locations (
            project_id, location_name, address, created_at
        ) VALUES (%s, %s, %s, %s)
        RETURNING location_id
        """
        
        self.cursor.execute(query, (
            project_id,
            location_info.get('location'),
            location_info.get('address'),
            datetime.now()
        ))
        
        location_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return location_id
    
    def insert_tbe_boq_file(self, project_id: int, boq_name: str, notes: str = None) -> int:
        query = """
        INSERT INTO to_be_estimated_boq_files (
            project_id, boq_name, created_at, updated_at, notes
        ) VALUES (%s, %s, %s, %s, %s)
        RETURNING boq_id
        """
        
        self.cursor.execute(query, (
            project_id,
            boq_name,
            datetime.now(),
            datetime.now(),
            notes
        ))
        
        boq_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return boq_id
    
    def insert_tbe_items_batch(self, boq_id: int, location_id: int, items: List[Dict]):
        if not items:
            return
        
        query = """
        INSERT INTO to_be_estimated_boq_items (
            boq_id, item_code, item_description, unit_of_measurement,
            quantity, location_id, created_at, updated_at
        ) VALUES %s
        """
        
        values = []
        for item in items:
            values.append((
                boq_id,
                item.get('item_code'),
                item['item_description'],
                item['unit_of_measurement'],
                item['quantity'],
                location_id,
                datetime.now(),
                datetime.now()
            ))
        
        try:
            execute_values(self.cursor, query, values)
            self.conn.commit()
            print(f"✓ Inserted {len(values)} items into to_be_estimated_boq_items")
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error inserting items: {e}")
            raise


class GeminiAgentSystem:
    """Gemini Agent for BoQ extraction"""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def extract_project_info(self, text: str) -> Dict:
        prompt = f"""
        Extract project information from this BoQ document.
        Return ONLY a JSON object:
        - project_name: string
        - project_code: string (generate if not found: PROJ-YYYY-XXX)
        - year: string (e.g., "2024" or "2024-2025")
        - location: string
        - client_name: string (optional)
        
        Text:
        {text[:2000]}
        
        Return only valid JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            return json.loads(result_text)
        except Exception as e:
            print(f"  Warning: Gemini extraction failed: {e}")
            return None
    
    def identify_columns(self, columns: List[str]) -> Dict:
        named_columns = [col for col in columns if not str(col).startswith('Unnamed')]
        
        prompt = f"""
        Analyze these Excel columns and map to BoQ fields (no pricing):
        
        Columns: {columns}
        Named: {named_columns}
        
        Map to:
        - item_code: Serial/item number
        - description: Item description
        - quantity: Quantity
        - unit: Unit of measurement
        
        Return ONLY JSON:
        {{
            "item_code": "column_name_or_null",
            "description": "column_name_or_null",
            "quantity": "column_name_or_null",
            "unit": "column_name_or_null"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(result_text)
            
            valid_result = {}
            for key, value in result.items():
                if value and value in columns:
                    valid_result[key] = value
                else:
                    valid_result[key] = None
            
            return valid_result
        except Exception as e:
            print(f"  Warning: Gemini column identification failed: {e}")
            return None


def extract_project_info_hybrid(df: pd.DataFrame, agent_system: GeminiAgentSystem) -> Dict:
    project_info = {
        'project_name': None,
        'project_code': None,
        'year': None,
        'location': None,
        'client_name': None
    }
    
    for idx, row in df.head(20).iterrows():
        for col in df.columns:
            cell_value = str(row[col]).strip() if pd.notna(row[col]) else ""
            cell_lower = cell_value.lower()
            
            if not project_info['project_name']:
                if any(keyword in cell_lower for keyword in ['project', 'work', 'contract']):
                    if len(cell_value) > 10 and len(cell_value) < 200:
                        project_info['project_name'] = cell_value
            
            if not project_info['location']:
                if any(keyword in cell_lower for keyword in ['location', 'site', 'place', 'district']):
                    next_col_idx = df.columns.get_loc(col) + 1
                    if next_col_idx < len(df.columns):
                        next_val = str(row[df.columns[next_col_idx]])
                        if pd.notna(row[df.columns[next_col_idx]]):
                            project_info['location'] = next_val
            
            if not project_info['year']:
                year_match = re.search(r'20\d{2}(?:-20?\d{2})?', cell_value)
                if year_match:
                    project_info['year'] = year_match.group()
    
    if not project_info['project_name'] or not project_info['location']:
        text = df.head(20).to_string()
        gemini_result = agent_system.extract_project_info(text)
        
        if gemini_result:
            for key in project_info.keys():
                if not project_info[key] and gemini_result.get(key):
                    project_info[key] = gemini_result[key]
    
    if not project_info['project_name']:
        project_info['project_name'] = f"Project {datetime.now().strftime('%Y%m%d')}"
    
    if not project_info['project_code']:
        year = project_info.get('year', datetime.now().year)
        project_info['project_code'] = f"PROJ-{year}-{datetime.now().strftime('%m%d%H%M')}"
    
    if not project_info['year']:
        project_info['year'] = str(datetime.now().year)
    
    if not project_info['location']:
        project_info['location'] = "Unknown Location"
    
    return project_info


def extract_boq_items_hybrid(df: pd.DataFrame, agent_system: GeminiAgentSystem,
                             use_parallel: bool = True, max_workers: int = 4) -> List[Dict]:
    print(f"  Analyzing DataFrame: {len(df)} rows, {len(df.columns)} columns")
    start_time = time.time()
    
    column_mapping = BoQPatternMatcher.identify_columns(df)
    print(f"  Initial column mapping: {column_mapping}")
    
    if not column_mapping['description'] or not column_mapping['quantity']:
        print(f"  Using Gemini for column identification...")
        gemini_mapping = agent_system.identify_columns(df.columns.tolist())
        
        if gemini_mapping:
            for key, value in gemini_mapping.items():
                if value and (not column_mapping.get(key) or column_mapping.get(key) is None):
                    column_mapping[key] = value
            
            print(f"  Updated column mapping: {column_mapping}")
    
    items = []
    for idx, row in df.iterrows():
        if column_mapping['description'] and pd.isna(row[column_mapping['description']]):
            continue
        
        is_item = False
        
        if column_mapping['item_code']:
            if BoQPatternMatcher.is_valid_item_code(row[column_mapping['item_code']]):
                is_item = True
        
        if column_mapping['quantity']:
            qty = BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']])
            if qty > 0:
                is_item = True
        
        if not is_item:
            continue
        
        item_code_val = row[column_mapping['item_code']] if column_mapping['item_code'] and pd.notna(row[column_mapping['item_code']]) else None
        item_code = str(item_code_val) if item_code_val is not None else None
        
        desc_val = row[column_mapping['description']] if column_mapping['description'] else ""
        description = str(desc_val) if pd.notna(desc_val) else ""
        
        item = {
            'item_code': item_code,
            'item_description': description,
            'quantity': BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']]) if column_mapping['quantity'] else 0,
            'unit_of_measurement': BoQPatternMatcher.normalize_unit(row[column_mapping['unit']]) if column_mapping['unit'] else "Each"
        }
        
        if item['quantity'] == 0:
            continue
        
        item['item_description'] = re.sub(r'\s+', ' ', item['item_description']).strip()
        items.append(item)
    
    elapsed = time.time() - start_time
    print(f"  Extracted {len(items)} items in {elapsed:.2f}s")
    
    return items


def process_tbe_boq_file(file_path: str, use_parallel: bool = True, max_workers: int = 4):
    """Process to-be-estimated BoQ file"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(file_path).name}")
    print(f"Mode: To-Be-Estimated (No Pricing)")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    agent_system = GeminiAgentSystem()
    db = BoQDatabase()
    db.connect()
    
    try:
        print("Step 1: Reading Excel file...")
        excel_file = pd.ExcelFile(file_path)
        sheets = {name: pd.read_excel(file_path, sheet_name=name) for name in excel_file.sheet_names}
        print(f"✓ Read {len(sheets)} sheets")
        
        print("\nStep 2: Extracting project information...")
        first_sheet = list(sheets.values())[0]
        project_info = extract_project_info_hybrid(first_sheet, agent_system)
        print(f"✓ Project: {project_info['project_name']}")
        print(f"  Code: {project_info['project_code']}")
        print(f"  Location: {project_info['location']}")
        
        print("\nStep 3: Inserting project...")
        project_id = db.insert_project(project_info)
        print(f"✓ Project ID: {project_id}")
        
        print("\nStep 4: Inserting location...")
        location_id = db.insert_location(project_id, project_info)
        print(f"✓ Location ID: {location_id}")
        
        print("\nStep 5: Creating to-be-estimated BoQ file record...")
        boq_name = Path(file_path).stem
        boq_id = db.insert_tbe_boq_file(
            project_id, 
            boq_name,
            notes=f"To-be-estimated BoQ uploaded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"✓ BoQ ID: {boq_id}")
        
        print("\nStep 6: Extracting BoQ items (quantities only)...")
        all_items = []
        
        for sheet_name, sheet_df in sheets.items():
            print(f"\n  Processing sheet: {sheet_name}")
            items = extract_boq_items_hybrid(sheet_df, agent_system, use_parallel, max_workers)
            all_items.extend(items)
        
        print(f"\nStep 7: Inserting {len(all_items)} items...")
        db.insert_tbe_items_batch(boq_id, location_id, all_items)
        
        elapsed_total = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Processing Complete!")
        print(f"{'='*60}")
        print(f"Project ID: {project_id}")
        print(f"Location ID: {location_id}")
        print(f"BoQ ID: {boq_id}")
        print(f"Total Items: {len(all_items)}")
        print(f"Processing Time: {elapsed_total:.2f}s")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'project_id': project_id,
            'location_id': location_id,
            'boq_id': boq_id,
            'total_items': len(all_items),
            'processing_time': elapsed_total
        }
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    finally:
        db.close()


def main():
    import sys
    
    if not os.getenv('GEMINI_API_KEY'):
        print("✗ Error: GEMINI_API_KEY not set")
        return
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = './estimate_input_files/sample.xlsx'
    
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        return
    
    result = process_tbe_boq_file(file_path, use_parallel=True, max_workers=4)
    
    if result['success']:
        print("\n✓ Success!")
    else:
        print(f"\n✗ Failed: {result.get('error')}")


if __name__ == "__main__":
    main()