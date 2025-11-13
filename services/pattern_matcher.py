"""
Pattern matching utilities for BOQ data extraction.
"""
import re
import pandas as pd
from typing import Dict, Optional


class PatternMatcher:
    """Pattern matching for BOQ column identification."""
    
    COLUMN_PATTERNS = {
        'item_code': [r'item[\s_-]*no', r'item[\s_-]*code', r's[\s_-]*no', r'sl[\s_-]*no'],
        'description': [r'description', r'particulars', r'scope[\s_-]*of[\s_-]*work'],
        'quantity': [r'qty', r'quantity', r'qnty'],
        'unit': [r'unit', r'uom', r'unit[\s_-]*of[\s_-]*measurement'],
        'supply_rate': [r'supply[\s_-]*rate', r'material[\s_-]*rate', r'supply[\s_-]*unit[\s_-]*rate', r'unit[\s_-]*rate'],
        'labour_rate': [r'labour[\s_-]*rate', r'labor[\s_-]*rate', r'labour[\s_-]*unit[\s_-]*rate', r'labor[\s_-]*unit[\s_-]*rate'],
        'supply_amount': [r'supply[\s_-]*amount', r'material[\s_-]*amount', r'supply[\s_-]*total'],
        'labour_amount': [r'labour[\s_-]*amount', r'labor[\s_-]*amount', r'labour[\s_-]*total', r'labor[\s_-]*total'],
        'total_amount': [r'total[\s_-]*amount', r'total', r'amount', r'grand[\s_-]*total'],
    }
    
    @classmethod
    def identify_columns(cls, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Identify columns in dataframe."""
        columns = {k: None for k in cls.COLUMN_PATTERNS.keys()}
        
        for col in df.columns:
            col_str = str(col).lower().strip()
            for field_type, patterns in cls.COLUMN_PATTERNS.items():
                if columns[field_type] is None:
                    for pattern in patterns:
                        if re.search(pattern, col_str, re.IGNORECASE):
                            columns[field_type] = col
                            break
        
        return columns
    
    @staticmethod
    def normalize_unit(unit: str) -> str:
        """Normalize unit of measurement."""
        if pd.isna(unit):
            return "Each"
        
        unit_str = str(unit).strip().lower()
        
        unit_map = {
            'sqm': 'Sqm', 'sq.m': 'Sqm', 'cum': 'Cum', 'cu.m': 'Cum',
            'mtr': 'Mtr', 'm': 'Mtr', 'kg': 'Kg', 'nos': 'Nos', 'no': 'Nos',
            'litre': 'Ltr', 'ltr': 'Ltr', 'ton': 'Ton', 'rmt': 'Rmt',
        }
        
        return unit_map.get(unit_str, unit_str.title())
    
    @staticmethod
    def extract_numeric(value) -> float:
        """Extract numeric value."""
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).replace(',', '')
        match = re.search(r'[-+]?[0-9]*\.?[0-9]+', value_str)
        return float(match.group()) if match else 0.0