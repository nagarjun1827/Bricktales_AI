import re
import pandas as pd
from typing import Dict, List, Optional

class TBEPatternMatcher:
    """Pattern matching for TBE BOQ data extraction - no pricing"""
    
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