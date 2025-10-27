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
        
        unit_lower = str(unit).lower().strip()
        
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
        
        return unit_map.get(unit_lower, unit.strip())
    
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
        
        self.cursor.execute("""
            INSERT INTO projects (project_name, project_code, client_name, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING project_id
        """, (
            project_info['project_name'],
            project_info['project_code'],
            project_info.get('client_name'),
            start_date,
            end_date
        ))
        project_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return project_id
    
    def insert_location(self, project_id: int, location_info: Dict) -> int:
        """Insert location and return location_id"""
        self.cursor.execute("""
            INSERT INTO locations (project_id, location_name, address)
            VALUES (%s, %s, %s)
            RETURNING location_id
        """, (
            project_id,
            location_info.get('location', 'Main Site'),
            location_info.get('location')
        ))
        location_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return location_id
    
    def insert_boq_file(self, project_id: int, file_info: Dict) -> int:
        """Insert BoQ file metadata"""
        self.cursor.execute("""
            INSERT INTO boq_files (
                project_id, file_name, file_path, file_type, 
                uploaded_by, version, notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING boq_id
        """, (
            project_id,
            file_info['file_name'],
            file_info.get('file_path'),
            file_info.get('file_type', 'xlsx'),
            file_info.get('uploaded_by', 'system'),
            file_info.get('version', 1),
            file_info.get('notes')
        ))
        boq_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return boq_id
    
    def insert_boq_items_batch(self, boq_id: int, location_id: int, items: List[Dict]):
        """Insert BoQ items in batch with better performance"""
        if not items:
            return
        
        values = []
        for item in items:
            try:
                values.append((
                    boq_id,
                    item.get('item_code'),
                    item['item_description'],
                    item['unit_of_measurement'],
                    float(item.get('quantity', 0)),
                    float(item.get('unit_rate', 0)) if item.get('unit_rate') else None,
                    None,  # labour_unit_rate
                    location_id
                ))
            except (ValueError, TypeError):
                continue
        
        if values:
            # Use execute_values for batch insert - much faster
            execute_values(
                self.cursor,
                """
                INSERT INTO boq_items (
                    boq_id, item_code, item_description, unit_of_measurement,
                    quantity, supply_unit_rate, labour_unit_rate, location_id
                )
                VALUES %s
                """,
                values,
                page_size=500  # Increased page size for better performance
            )
            self.conn.commit()


class GeminiAgentSystem:
    """Gemini-powered Agent System following OpenAI Agents SDK structure"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini agent system"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment")
        
        genai.configure(api_key=self.api_key)
        
        # Create model instances for different agents
        self.project_agent_model = genai.GenerativeModel('gemini-2.5-flash')
        self.column_mapper_model = genai.GenerativeModel('gemini-2.5-flash')
        self.batch_extractor_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Agent instructions
        self.project_agent_instructions = """You are an expert at extracting project information from construction Bill of Quantities (BoQ) documents.

Extract:
1. Project Name - the full name of the construction project
2. Project Code - any project identifier or code (if not found, generate from first 3 words)
3. Client Name - the client or organization name
4. Location - the project location or address
5. Year - the project year or year range (e.g., 2020-2021)

ALWAYS return ONLY valid JSON:
{
    "project_name": "...",
    "project_code": "...",
    "client_name": "...",
    "location": "...",
    "year": "..."
}

Use null for fields not found. No additional text."""

        self.column_mapper_instructions = """You are an expert at identifying column structures in BoQ spreadsheets.

Identify which columns contain:
1. Item Code/Number (e.g., 1.1, 2.3, A1)
2. Description (work description)
3. Quantity
4. Unit of Measurement
5. Rate (unit rate/price)
6. Amount (total amount)

ALWAYS return ONLY valid JSON:
{
    "item_code_column": "exact_column_name",
    "description_column": "exact_column_name",
    "quantity_column": "exact_column_name",
    "unit_column": "exact_column_name",
    "rate_column": "exact_column_name",
    "amount_column": "exact_column_name"
}

Use exact column names. Use null for columns not found. No additional text."""

        self.batch_extractor_instructions = """You are an expert at extracting BoQ line items from tabular data.

Extract ALL BoQ items from the provided data chunk.

Return ONLY a valid JSON array:
[
    {
        "item_code": "1.1",
        "item_description": "Full description",
        "quantity": 10.5,
        "unit_of_measurement": "Sqm",
        "unit_rate": 100.0,
        "amount": 1050.0
    }
]

Rules:
- Skip header rows and section titles
- Extract only actual work items with quantities > 0
- Convert all numeric values properly
- Standardize units (Sqm, Cum, Mtr, Nos, Each, Kg, etc.)
- Clean descriptions (remove extra spaces)
- Return ONLY the JSON array, no additional text"""
    
    def extract_project_info(self, sample_data: str) -> Dict:
        """Extract project information using Gemini agent"""
        try:
            prompt = f"{self.project_agent_instructions}\n\nData:\n{sample_data}"
            response = self.project_agent_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
        except Exception as e:
            print(f"  Gemini agent error: {e}")
            return None
    
    def identify_columns(self, df_sample: str, columns_list: List[str]) -> Dict:
        """Identify column mappings using Gemini agent"""
        try:
            prompt = f"{self.column_mapper_instructions}\n\nColumns: {columns_list}\n\nSample:\n{df_sample}"
            response = self.column_mapper_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
        except Exception as e:
            print(f"  Gemini agent error: {e}")
            return None
    
    def extract_items_batch(self, data_chunk: str, chunk_id: int) -> List[Dict]:
        """Extract items from a data chunk using Gemini"""
        try:
            prompt = f"{self.batch_extractor_instructions}\n\nData Chunk {chunk_id}:\n{data_chunk}"
            response = self.batch_extractor_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            items = json.loads(response_text.strip())
            return items if isinstance(items, list) else []
        except Exception as e:
            print(f"  Gemini batch extraction error (chunk {chunk_id}): {e}")
            return []


def extract_project_info_hybrid(df: pd.DataFrame, agent_system: GeminiAgentSystem) -> Dict:
    """Extract project information using hybrid approach"""
    
    # First, try regex pattern matching
    project_info = {
        'project_name': None,
        'project_code': None,
        'client_name': None,
        'location': None,
        'year': None
    }
    
    # Search first 10 rows for project information
    for idx, row in df.head(10).iterrows():
        row_text = ' '.join([str(val) for val in row.values if pd.notna(val)])
        
        # Extract year patterns (e.g., 2020-2021, 2020)
        if not project_info['year']:
            year_match = re.search(r'(20\d{2}[-–]20\d{2}|20\d{2})', row_text)
            if year_match:
                project_info['year'] = year_match.group(1)
        
        # Look for project name (usually in first row with long text)
        if not project_info['project_name'] and len(row_text) > 50:
            clean_text = re.sub(r'\s+', ' ', row_text).strip()
            if 'bill of quantities' in clean_text.lower() or 'boq' in clean_text.lower():
                project_info['project_name'] = clean_text
        
        # Look for location keywords
        if not project_info['location']:
            location_keywords = ['location', 'site', 'address', 'place', 'kolkata', 'mumbai', 'delhi', 'kasba']
            for keyword in location_keywords:
                if keyword in row_text.lower():
                    project_info['location'] = row_text
                    break
    
    # If regex didn't find everything, use Gemini Agent
    if not all([project_info['project_name'], project_info['year']]):
        print("  Using Gemini Agent for project info extraction...")
        sample_data = df.head(10).to_string()
        
        agent_info = agent_system.extract_project_info(sample_data)
        
        if agent_info:
            for key in project_info:
                if not project_info[key] and agent_info.get(key):
                    project_info[key] = agent_info[key]
    
    # Generate project code if not found
    if not project_info['project_code']:
        if project_info['project_name']:
            code_parts = project_info['project_name'].split()[:3]
            project_info['project_code'] = '_'.join(code_parts).upper().replace(',', '').replace('.', '')
        else:
            project_info['project_code'] = f"PROJ_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Set default project name if still not found
    if not project_info['project_name']:
        project_info['project_name'] = "Unknown Project"
    
    return project_info


def split_dataframe_into_chunks(df: pd.DataFrame, chunk_size: int = 50) -> List[Tuple[int, pd.DataFrame]]:
    """Split DataFrame into smaller chunks for parallel processing"""
    chunks = []
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx]
        chunks.append((i, chunk_df))
    
    return chunks


def extract_boq_items_parallel(df: pd.DataFrame, agent_system: GeminiAgentSystem, 
                               column_mapping: Dict, max_workers: int = 4) -> List[Dict]:
    """Extract BoQ items using parallel processing with Gemini"""
    
    print(f"  Using parallel processing with {max_workers} workers...")
    
    # First, try regex-based extraction (fast path)
    regex_items = []
    gemini_chunks = []
    
    # Split into chunks
    chunks = split_dataframe_into_chunks(df, chunk_size=50)
    
    for chunk_id, chunk_df in chunks:
        chunk_items = []
        needs_gemini = False
        
        for idx, row in chunk_df.iterrows():
            # Skip if no description
            if column_mapping['description'] and pd.isna(row[column_mapping['description']]):
                continue
            
            # Check if this is a valid item
            is_item = False
            
            if column_mapping['item_code']:
                if BoQPatternMatcher.is_valid_item_code(row[column_mapping['item_code']]):
                    is_item = True
            
            if column_mapping['quantity']:
                qty = BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']])
                if qty > 0:
                    is_item = True
            
            if not is_item:
                needs_gemini = True
                continue
            
            # Extract item data using regex
            item = {
                'item_code': str(row[column_mapping['item_code']]) if column_mapping['item_code'] and pd.notna(row[column_mapping['item_code']]) else None,
                'item_description': str(row[column_mapping['description']]) if column_mapping['description'] else "",
                'quantity': BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']]) if column_mapping['quantity'] else 0,
                'unit_of_measurement': BoQPatternMatcher.normalize_unit(row[column_mapping['unit']]) if column_mapping['unit'] else "Each",
                'unit_rate': BoQPatternMatcher.extract_numeric(row[column_mapping['rate']]) if column_mapping['rate'] else 0,
                'amount': BoQPatternMatcher.extract_numeric(row[column_mapping['amount']]) if column_mapping['amount'] else 0
            }
            
            if item['quantity'] > 0:
                item['item_description'] = re.sub(r'\s+', ' ', item['item_description']).strip()
                chunk_items.append(item)
        
        regex_items.extend(chunk_items)
        
        # If regex failed for this chunk, mark for Gemini processing
        if needs_gemini and len(chunk_items) < len(chunk_df) * 0.5:
            gemini_chunks.append((chunk_id, chunk_df))
    
    print(f"  Regex extracted: {len(regex_items)} items")
    
    # If we have chunks that need Gemini processing, do it in parallel
    if gemini_chunks:
        print(f"  Processing {len(gemini_chunks)} chunks with Gemini in parallel...")
        
        gemini_items = []
        
        def process_chunk_with_gemini(chunk_tuple):
            chunk_id, chunk_df = chunk_tuple
            chunk_str = chunk_df.to_string()
            return agent_system.extract_items_batch(chunk_str, chunk_id)
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk_with_gemini, chunk): chunk 
                             for chunk in gemini_chunks}
            
            for future in as_completed(future_to_chunk):
                try:
                    items = future.result()
                    gemini_items.extend(items)
                except Exception as e:
                    print(f"  Error processing chunk: {e}")
        
        print(f"  Gemini extracted: {len(gemini_items)} additional items")
        
        # Combine results
        all_items = regex_items + gemini_items
    else:
        all_items = regex_items
    
    return all_items


def extract_boq_items_hybrid(df: pd.DataFrame, agent_system: GeminiAgentSystem, 
                            use_parallel: bool = True, max_workers: int = 4) -> List[Dict]:
    """Extract BoQ items using hybrid approach with optional parallel processing"""
    
    start_time = time.time()
    
    # Step 1: Use regex pattern matching to identify columns
    column_mapping = BoQPatternMatcher.identify_columns(df)
    
    print(f"  Column mapping (regex): {column_mapping}")
    
    # If pattern matching failed to identify key columns, use Gemini Agent
    if not column_mapping['description'] or not column_mapping['quantity']:
        print("  Using Gemini Agent for column identification...")
        
        sample_data = df.head(5).to_string()
        columns_list = list(df.columns)
        
        agent_mapping = agent_system.identify_columns(sample_data, columns_list)
        
        if agent_mapping:
            for key, col_key in [
                ('item_code', 'item_code_column'),
                ('description', 'description_column'),
                ('quantity', 'quantity_column'),
                ('unit', 'unit_column'),
                ('rate', 'rate_column'),
                ('amount', 'amount_column')
            ]:
                if not column_mapping[key] and agent_mapping.get(col_key):
                    column_mapping[key] = agent_mapping[col_key]
            
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
            
            item = {
                'item_code': str(row[column_mapping['item_code']]) if column_mapping['item_code'] and pd.notna(row[column_mapping['item_code']]) else None,
                'item_description': str(row[column_mapping['description']]) if column_mapping['description'] else "",
                'quantity': BoQPatternMatcher.extract_numeric(row[column_mapping['quantity']]) if column_mapping['quantity'] else 0,
                'unit_of_measurement': BoQPatternMatcher.normalize_unit(row[column_mapping['unit']]) if column_mapping['unit'] else "Each",
                'unit_rate': BoQPatternMatcher.extract_numeric(row[column_mapping['rate']]) if column_mapping['rate'] else 0,
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
        boq_id = db.insert_boq_file(project_id, file_info)
        print(f"✓ File ID: {boq_id}")
        
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
        db.insert_boq_items_batch(boq_id, location_id, all_items)
        print(f"✓ Inserted {len(all_items)} items")
        
        # Summary
        elapsed_total = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Processing Complete!")
        print(f"{'='*60}")
        print(f"Project ID: {project_id}")
        print(f"Location ID: {location_id}")
        print(f"File ID: {boq_id}")
        print(f"Total Items: {len(all_items)}")
        
        # Calculate totals
        total_amount = sum(item.get('amount', 0) for item in all_items)
        print(f"Total Amount: ₹{total_amount:,.2f}")
        print(f"Processing Time: {elapsed_total:.2f}s")
        print(f"Items per second: {len(all_items)/elapsed_total:.2f}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'project_id': project_id,
            'location_id': location_id,
            'boq_id': boq_id,
            'total_items': len(all_items),
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
        file_path = './input_files/boq_3.xlsx'
    
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        return
    
    # Configuration
    use_parallel = True  # Enable parallel processing
    max_workers = 4      # Number of parallel workers
    
    # Process file
    result = process_boq_file_with_agents(file_path, use_parallel=use_parallel, max_workers=max_workers)
    
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