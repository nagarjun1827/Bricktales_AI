import os
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import partial
import time

# Load environment variables
load_dotenv()


class BoQPatternMatcher:
    """Pattern matching utilities for BoQ data extraction"""
    
    # Column name patterns for matching
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
        ],
        'rate': [
            r'rate',
            r'unit[\s_-]*rate',
            r'price',
            r'rate[\s_-]*\([\s_-]*rs',
        ],
        'supply_rate': [
            r'supply[\s_-]*rate',
            r'material[\s_-]*rate',
            r'supply[\s_-]*unit[\s_-]*rate',
        ],
        'labour_rate': [
            r'labour[\s_-]*rate',
            r'labor[\s_-]*rate',
            r'labour[\s_-]*unit[\s_-]*rate',
            r'labor[\s_-]*unit[\s_-]*rate',
        ],
        'amount': [
            r'amount',
            r'total',
            r'value',
            r'amount[\s_-]*\([\s_-]*rs',
        ]
    }
    
    # Item code patterns (e.g., 1.1, 2.3.4, A1, etc.)
    ITEM_CODE_PATTERN = re.compile(r'^[A-Z]?[0-9]+(\.[0-9]+)*$')
    
    @staticmethod
    def normalize_column_name(col_name: str) -> str:
        """Normalize column name for matching"""
        if pd.isna(col_name):
            return ""
        return str(col_name).lower().strip()
    
    @classmethod
    def match_column(cls, col_name: str, field_type: str) -> bool:
        """Check if column name matches a field type"""
        normalized = cls.normalize_column_name(col_name)
        patterns = cls.COLUMN_PATTERNS.get(field_type, [])
        
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def identify_columns(cls, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Identify column mappings in the DataFrame"""
        columns = {
            'item_code': None,
            'description': None,
            'quantity': None,
            'unit': None,
            'rate': None,
            'supply_rate': None,
            'labour_rate': None,
            'amount': None
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
        """Check if value looks like a valid item code"""
        if pd.isna(value):
            return False
        value_str = str(value).strip()
        return bool(cls.ITEM_CODE_PATTERN.match(value_str))
    
    @classmethod
    def normalize_unit(cls, unit: str) -> str:
        """Normalize unit of measurement"""
        if pd.isna(unit):
            return "Each"
        
        # Convert to string first to handle integers/floats
        unit_str = str(unit).strip()
        
        # If it's just a number, return "Each"
        if unit_str.replace('.', '').replace('-', '').isdigit():
            return "Each"
        
        unit_lower = unit_str.lower()
        
        # Map common variations to standard units
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
        """Extract numeric value from string"""
        if pd.isna(value):
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Remove commas and extract numbers
        value_str = str(value).replace(',', '')
        match = re.search(r'[-+]?[0-9]*\.?[0-9]+', value_str)
        
        if match:
            try:
                return float(match.group())
            except ValueError:
                return 0.0
        return 0.0


class BoQDatabase:
    """Database operations for BoQ storage"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        if db_config is None:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'boq_database'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD','root'),
                'port': int(os.getenv('DB_PORT', 5432))
            }
        else:
            self.db_config = db_config
        
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to database"""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def insert_project(self, project_info: Dict) -> int:
        """Insert project and return project_id"""
        # Parse year - handle ranges like "2020-2021"
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
        """Insert location and return location_id"""
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
    
    def insert_boq_file(self, project_id: int, file_info: Dict) -> int:
        """Insert BOQ file metadata and return file_id"""
        query = """
        INSERT INTO store_boq_files (
            project_id, file_name, file_path, file_type,
            upload_timestamp, uploaded_by, version, is_active, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING file_id
        """
        
        self.cursor.execute(query, (
            project_id,
            file_info['file_name'],
            file_info.get('file_path'),
            file_info.get('file_type', 'xlsx'),
            datetime.now(),
            file_info.get('uploaded_by', 'system'),
            file_info.get('version', 1),
            file_info.get('is_active', True),
            file_info.get('notes')
        ))
        
        file_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return file_id
    
    def insert_boq_items_batch(self, file_id: int, location_id: int, items: List[Dict]):
        """Batch insert BOQ items - FIXED VERSION"""
        if not items:
            return
        
        query = """
        INSERT INTO store_boq_items (
            file_id, item_code, item_description, unit_of_measurement,
            quantity, supply_unit_rate, labour_unit_rate, 
            location_id, created_at, updated_at
        ) VALUES %s
        """
        
        values = []
        for item in items:
            # Handle different rate scenarios
            supply_rate = item.get('supply_unit_rate', 0)
            labour_rate = item.get('labour_unit_rate', 0)
            
            # If only 'unit_rate' is provided, treat it as supply_rate
            if supply_rate == 0 and labour_rate == 0 and item.get('unit_rate'):
                supply_rate = item['unit_rate']
            
            # FIX: Use 0 instead of None for rates to satisfy CHECK constraints
            # The CHECK constraint requires >= 0, and NULL fails this
            # Also, generated columns need non-NULL values to compute
            values.append((
                file_id,
                item.get('item_code'),
                item['item_description'],
                item['unit_of_measurement'],
                item['quantity'],
                supply_rate,  # Changed from: supply_rate if supply_rate > 0 else None
                labour_rate,  # Changed from: labour_rate if labour_rate > 0 else None
                location_id,
                datetime.now(),
                datetime.now()
            ))
        
        try:
            execute_values(self.cursor, query, values)
            self.conn.commit()
            print(f"✓ Successfully inserted {len(values)} items into store_boq_items")
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error inserting items: {e}")
            print(f"Sample item data: {values[0] if values else 'No items'}")
            raise


class GeminiAgentSystem:
    """Gemini Agent System for BoQ extraction"""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def extract_project_info(self, text: str) -> Dict:
        """Extract project information using Gemini"""
        prompt = f"""
        Extract project information from this BoQ document text.
        Return ONLY a JSON object with these fields:
        - project_name: string
        - project_code: string (generate if not found, format: PROJ-YYYY-XXX)
        - year: string (e.g., "2024" or "2024-2025")
        - location: string
        - client_name: string (optional)
        
        Text:
        {text[:2000]}
        
        Return only valid JSON, no other text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            return json.loads(result_text)
        except Exception as e:
            print(f"  Warning: Gemini extraction failed: {e}")
            return None
    
    def identify_columns(self, columns: List[str]) -> Dict:
        """Identify column mappings using Gemini"""
        # Filter out Unnamed columns for better analysis
        named_columns = [col for col in columns if not str(col).startswith('Unnamed')]
        
        prompt = f"""
        Analyze these Excel column headers and identify which ones correspond to standard Bill of Quantities (BoQ) fields.
        
        Available columns: {columns}
        Named columns: {named_columns}
        
        Map these fields to the appropriate column names (use exact column name from the list):
        - item_code: Serial number, item number, or code (often first column)
        - description: Item description or work description (often second column, usually longest text)
        - quantity: Quantity or qty
        - unit: Unit of measurement (UOM)
        - supply_rate: Supply/material rate per unit
        - labour_rate: Labour rate per unit
        - rate: Combined rate (if supply and labour not separate)
        - amount: Total amount or value
        
        Return ONLY a valid JSON object like this:
        {{
            "item_code": "column_name_or_null",
            "description": "column_name_or_null",
            "quantity": "column_name_or_null",
            "unit": "column_name_or_null",
            "supply_rate": null,
            "labour_rate": null,
            "rate": "column_name_or_null",
            "amount": "column_name_or_null"
        }}
        
        Use null for fields not found. Return ONLY the JSON, no explanation.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(result_text)
            
            # Validate that returned column names actually exist
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
    """Extract project info using hybrid approach (pattern + Gemini)"""
    
    # Try pattern-based extraction first
    project_info = {
        'project_name': None,
        'project_code': None,
        'year': None,
        'location': None,
        'client_name': None
    }
    
    # Search for project info in first 20 rows
    for idx, row in df.head(20).iterrows():
        for col in df.columns:
            cell_value = str(row[col]).strip() if pd.notna(row[col]) else ""
            cell_lower = cell_value.lower()
            
            # Project name patterns
            if not project_info['project_name']:
                if any(keyword in cell_lower for keyword in ['project', 'work', 'contract']):
                    if len(cell_value) > 10 and len(cell_value) < 200:
                        project_info['project_name'] = cell_value
            
            # Location patterns
            if not project_info['location']:
                if any(keyword in cell_lower for keyword in ['location', 'site', 'place', 'district']):
                    next_col_idx = df.columns.get_loc(col) + 1
                    if next_col_idx < len(df.columns):
                        next_val = str(row[df.columns[next_col_idx]])
                        if pd.notna(row[df.columns[next_col_idx]]):
                            project_info['location'] = next_val
            
            # Year patterns
            if not project_info['year']:
                year_match = re.search(r'20\d{2}(?:-20?\d{2})?', cell_value)
                if year_match:
                    project_info['year'] = year_match.group()
    
    # Fallback to Gemini if needed
    if not project_info['project_name'] or not project_info['location']:
        text = df.head(20).to_string()
        gemini_result = agent_system.extract_project_info(text)
        
        if gemini_result:
            for key in project_info.keys():
                if not project_info[key] and gemini_result.get(key):
                    project_info[key] = gemini_result[key]
    
    # Generate defaults if still missing
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


def extract_boq_items_parallel(df: pd.DataFrame, agent_system: GeminiAgentSystem, 
                                column_mapping: Dict, max_workers: int = 4) -> List[Dict]:
    """Extract items in parallel using thread pool"""
    
    def process_row(row_data):
        idx, row = row_data
        
        if column_mapping['description'] and pd.isna(row[column_mapping['description']]):
            return None
        
        is_item = False
        
        if column_mapping['item_code']:
            if BoQPatternMatcher.is_valid_item_code(row[column_mapping['item_code']]):
                is_item = True
        
        if column_mapping['quantity']:
            qty = BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']])
            if qty > 0:
                is_item = True
        
        if not is_item:
            return None
        
        # Extract supply and labour rates separately if available
        supply_rate = 0
        labour_rate = 0
        
        if column_mapping.get('supply_rate'):
            supply_rate = BoQPatternMatcher.extract_numeric(row[column_mapping['supply_rate']])
        
        if column_mapping.get('labour_rate'):
            labour_rate = BoQPatternMatcher.extract_numeric(row[column_mapping['labour_rate']])
        
        # Fallback to combined rate
        if supply_rate == 0 and labour_rate == 0 and column_mapping.get('rate'):
            supply_rate = BoQPatternMatcher.extract_numeric(row[column_mapping['rate']])
        
        # Safely convert item_code to string
        item_code_val = row[column_mapping['item_code']] if column_mapping['item_code'] and pd.notna(row[column_mapping['item_code']]) else None
        item_code = str(item_code_val) if item_code_val is not None else None
        
        # Safely convert description to string
        desc_val = row[column_mapping['description']] if column_mapping['description'] else ""
        description = str(desc_val) if pd.notna(desc_val) else ""
        
        item = {
            'item_code': item_code,
            'item_description': description,
            'quantity': BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']]) if column_mapping['quantity'] else 0,
            'unit_of_measurement': BoQPatternMatcher.normalize_unit(row[column_mapping['unit']]) if column_mapping['unit'] else "Each",
            'supply_unit_rate': supply_rate,
            'labour_unit_rate': labour_rate,
            'amount': BoQPatternMatcher.extract_numeric(row[column_mapping['amount']]) if column_mapping['amount'] else 0
        }
        
        if item['quantity'] == 0:
            return None
        
        item['item_description'] = re.sub(r'\s+', ' ', item['item_description']).strip()
        return item
    
    items = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, (idx, row)) for idx, row in df.iterrows()]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    items.append(result)
            except Exception as e:
                print(f"    Error processing row: {e}")
    
    return items


def extract_boq_items_hybrid(df: pd.DataFrame, agent_system: GeminiAgentSystem,
                             use_parallel: bool = True, max_workers: int = 4) -> List[Dict]:
    """Extract BoQ items using hybrid pattern matching + Gemini"""
    
    print(f"  Analyzing DataFrame: {len(df)} rows, {len(df.columns)} columns")
    start_time = time.time()
    
    # Step 1: Identify columns (pattern first, then Gemini if needed)
    column_mapping = BoQPatternMatcher.identify_columns(df)
    print(f"  Initial column mapping (patterns): {column_mapping}")
    
    # Use Gemini to improve mapping if any critical fields are missing
    if not column_mapping['description'] or not column_mapping['quantity']:
        print(f"  Using Gemini to identify columns...")
        gemini_mapping = agent_system.identify_columns(df.columns.tolist())
        
        if gemini_mapping:
            for key, value in gemini_mapping.items():
                if value and (not column_mapping.get(key) or column_mapping.get(key) is None):
                    column_mapping[key] = value
            
            print(f"  Updated column mapping (Gemini): {column_mapping}")
    
    # Step 2: Extract items
    if use_parallel and len(df) > 100:
        items = extract_boq_items_parallel(df, agent_system, column_mapping, max_workers)
    else:
        # Sequential extraction for small datasets
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
            
            # Extract supply and labour rates separately if available
            supply_rate = 0
            labour_rate = 0
            
            if column_mapping.get('supply_rate'):
                supply_rate = BoQPatternMatcher.extract_numeric(row[column_mapping['supply_rate']])
            
            if column_mapping.get('labour_rate'):
                labour_rate = BoQPatternMatcher.extract_numeric(row[column_mapping['labour_rate']])
            
            # Fallback to combined rate
            if supply_rate == 0 and labour_rate == 0 and column_mapping.get('rate'):
                supply_rate = BoQPatternMatcher.extract_numeric(row[column_mapping['rate']])
            
            # Safely convert item_code to string
            item_code_val = row[column_mapping['item_code']] if column_mapping['item_code'] and pd.notna(row[column_mapping['item_code']]) else None
            item_code = str(item_code_val) if item_code_val is not None else None
            
            # Safely convert description to string
            desc_val = row[column_mapping['description']] if column_mapping['description'] else ""
            description = str(desc_val) if pd.notna(desc_val) else ""
            
            item = {
                'item_code': item_code,
                'item_description': description,
                'quantity': BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']]) if column_mapping['quantity'] else 0,
                'unit_of_measurement': BoQPatternMatcher.normalize_unit(row[column_mapping['unit']]) if column_mapping['unit'] else "Each",
                'supply_unit_rate': supply_rate,
                'labour_unit_rate': labour_rate,
                'amount': BoQPatternMatcher.extract_numeric(row[column_mapping['amount']]) if column_mapping['amount'] else 0
            }
            
            if item['quantity'] == 0:
                continue
            
            item['item_description'] = re.sub(r'\s+', ' ', item['item_description']).strip()
            items.append(item)
    
    elapsed = time.time() - start_time
    print(f"  Extracted {len(items)} items in {elapsed:.2f}s")
    
    return items


def process_boq_file_with_agents(file_path: str, uploaded_by: str = 'system', 
                                 use_parallel: bool = True, max_workers: int = 4):
    """Process BoQ file using Gemini-powered Agents SDK with parallel processing"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(file_path).name}")
    print(f"Using Gemini 2.0 Flash with OpenAI Agents SDK Structure")
    print(f"Parallel Processing: {'Enabled' if use_parallel else 'Disabled'}")
    if use_parallel:
        print(f"Max Workers: {max_workers}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Initialize Gemini Agent System
    agent_system = GeminiAgentSystem()
    
    # Initialize database
    db = BoQDatabase()
    db.connect()
    
    try:
        # Step 1: Read Excel file
        print("Step 1: Reading Excel file...")
        excel_file = pd.ExcelFile(file_path)
        sheets = {name: pd.read_excel(file_path, sheet_name=name) for name in excel_file.sheet_names}
        print(f"✓ Read {len(sheets)} sheets")
        
        # Step 2: Extract project information
        print("\nStep 2: Extracting project information...")
        first_sheet = list(sheets.values())[0]
        project_info = extract_project_info_hybrid(first_sheet, agent_system)
        print(f"✓ Project: {project_info['project_name']}")
        print(f"  Code: {project_info['project_code']}")
        print(f"  Year: {project_info['year']}")
        print(f"  Location: {project_info['location']}")
        
        # Step 3: Insert project
        print("\nStep 3: Inserting project into database...")
        project_id = db.insert_project(project_info)
        print(f"✓ Project ID: {project_id}")
        
        # Step 4: Insert location
        print("\nStep 4: Inserting location...")
        location_id = db.insert_location(project_id, project_info)
        print(f"✓ Location ID: {location_id}")
        
        # Step 5: Insert file metadata
        print("\nStep 5: Inserting file metadata...")
        file_info = {
            'file_name': Path(file_path).name,
            'file_path': file_path,
            'file_type': 'xlsx',
            'uploaded_by': uploaded_by,
            'version': 1,
            'notes': f"Processed with Gemini Agents SDK on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        file_id = db.insert_boq_file(project_id, file_info)
        print(f"✓ File ID: {file_id}")
        
        # Step 6: Extract and insert items from all sheets (parallel processing)
        print("\nStep 6: Extracting BoQ items from sheets...")
        all_items = []
        
        # Process sheets in parallel if multiple sheets
        if use_parallel and len(sheets) > 1:
            print(f"  Processing {len(sheets)} sheets in parallel...")
            
            def process_sheet(sheet_tuple):
                sheet_name, sheet_df = sheet_tuple
                print(f"\n  Processing sheet: {sheet_name}")
                return extract_boq_items_hybrid(sheet_df, agent_system, use_parallel, max_workers)
            
            with ThreadPoolExecutor(max_workers=min(len(sheets), max_workers)) as executor:
                future_to_sheet = {executor.submit(process_sheet, item): item 
                                 for item in sheets.items()}
                
                for future in as_completed(future_to_sheet):
                    try:
                        items = future.result()
                        all_items.extend(items)
                    except Exception as e:
                        print(f"  Error processing sheet: {e}")
        else:
            # Sequential processing
            for sheet_name, sheet_df in sheets.items():
                print(f"\n  Processing sheet: {sheet_name}")
                items = extract_boq_items_hybrid(sheet_df, agent_system, use_parallel, max_workers)
                all_items.extend(items)
        
        # Step 7: Insert items (batch insert)
        print(f"\nStep 7: Inserting {len(all_items)} items...")
        db.insert_boq_items_batch(file_id, location_id, all_items)
        print(f"✓ Inserted {len(all_items)} items")
        
        # Summary
        elapsed_total = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Processing Complete!")
        print(f"{'='*60}")
        print(f"Project ID: {project_id}")
        print(f"Location ID: {location_id}")
        print(f"File ID: {file_id}")
        print(f"Total Items: {len(all_items)}")
        
        # Calculate totals (supply + labour)
        total_supply = sum(item.get('supply_unit_rate', 0) * item.get('quantity', 0) for item in all_items)
        total_labour = sum(item.get('labour_unit_rate', 0) * item.get('quantity', 0) for item in all_items)
        total_amount = total_supply + total_labour
        
        print(f"Total Supply Amount: ₹{total_supply:,.2f}")
        print(f"Total Labour Amount: ₹{total_labour:,.2f}")
        print(f"Total Amount: ₹{total_amount:,.2f}")
        print(f"Processing Time: {elapsed_total:.2f}s")
        print(f"Items per second: {len(all_items)/elapsed_total:.2f}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'project_id': project_id,
            'location_id': location_id,
            'file_id': file_id,
            'total_items': len(all_items),
            'total_supply': total_supply,
            'total_labour': total_labour,
            'total_amount': total_amount,
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
    """Main execution"""
    import sys
    
    # Check for Gemini API key
    if not os.getenv('GEMINI_API_KEY'):
        print("✗ Error: GEMINI_API_KEY environment variable not set")
        print("  Set it using: export GEMINI_API_KEY='your-api-key'")
        return
    
    # Get file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = './input_files/30_40.xlsx'
    
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        return
    
    # Configuration
    use_parallel = True  # Enable parallel processing
    max_workers = 4      # Number of parallel workers
    
    # Process file
    result = process_boq_file_with_agents(
        file_path, 
        use_parallel=use_parallel, 
        max_workers=max_workers
    )
    
    if result['success']:
        print("\n✓ Success!")
        print(f"\nPerformance:")
        print(f"  Total Items: {result['total_items']}")
        print(f"  Processing Time: {result['processing_time']:.2f}s")
        print(f"  Throughput: {result['total_items']/result['processing_time']:.2f} items/s")
    else:
        print(f"\n✗ Failed: {result.get('error')}")


if __name__ == "__main__":
    main()