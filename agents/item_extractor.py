"""
Item extraction agent for BOQ processing.
"""
import logging
import pandas as pd
import re
from typing import List, Optional, Any
from models.store_boq_models import BOQItem

logger = logging.getLogger(__name__)


class ItemExtractorAgent:
    """Agent responsible for extracting BOQ items from Excel sheets."""
    
    def execute(
        self, 
        df: pd.DataFrame, 
        structure: dict, 
        boq_id: int, 
        location_id: int
    ) -> List[BOQItem]:
        """Extract BOQ items from dataframe."""
        logger.info("Extracting items with intelligent parsing...")
        
        data_start = structure['data_start_row']
        column_structure = structure['column_structure']
        
        col_map = {col['type']: col['position'] for col in column_structure}
        logger.debug(f"Column mapping: {col_map}")
        
        items = []
        data_df = df.iloc[data_start:].reset_index(drop=True)
        
        for _, row in data_df.iterrows():
            if row.isna().all():
                continue
            
            # Extract values
            item_code = self._get_value(row, col_map.get('item_code'))
            description = self._get_value(row, col_map.get('description'), default='')
            unit = self._get_value(row, col_map.get('unit'), default='Each')
            quantity = self._extract_numeric(self._get_value(row, col_map.get('quantity')))
            rate = self._extract_numeric(self._get_value(row, col_map.get('rate')))
            amount = self._extract_numeric(self._get_value(row, col_map.get('amount')))
            
            # Skip invalid rows
            if not description or (quantity == 0 and rate == 0 and amount == 0):
                continue
            if len(str(description).strip()) < 5:
                continue
            
            # Derive quantity if missing but have rate and amount
            if quantity == 0 and rate > 0 and amount > 0:
                quantity = round(amount / rate, 4)
            
            # Derive rate if missing but have quantity and amount
            if rate == 0 and quantity > 0 and amount > 0:
                rate = round(amount / quantity, 2)
            
            # Create item
            item = BOQItem(
                boq_id=boq_id,
                item_code=str(item_code) if item_code is not None else None,
                item_description=str(description).strip(),
                unit_of_measurement=self._normalize_unit(unit),
                quantity=quantity,
                supply_unit_rate=rate,
                labour_unit_rate=0.0,
                location_id=location_id,
                created_by='system'
            )
            
            items.append(item)
        
        logger.info(f"Extracted {len(items)} valid items")
        
        # Calculate summary
        total_supply = sum(item.supply_amount for item in items)
        total_labour = sum(item.labour_amount for item in items)
        total_amount = sum(item.total_amount for item in items)
        
        logger.info(f"Supply Total: ₹{total_supply:,.2f}")
        logger.info(f"Labour Total: ₹{total_labour:,.2f}")
        logger.info(f"Grand Total: ₹{total_amount:,.2f}")
        
        return items
    
    @staticmethod
    def _get_value(row: pd.Series, position: Optional[int], default=None) -> Any:
        """Get value from row at specified position."""
        if position is None or position >= len(row):
            return default
        val = row.iloc[position]
        return val if pd.notna(val) else default
    
    @staticmethod
    def _extract_numeric(value) -> float:
        """Extract numeric value from mixed content."""
        if value is None or pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).replace(',', '').strip()
        match = re.search(r'[-+]?[0-9]*\.?[0-9]+', value_str)
        return float(match.group()) if match else 0.0
    
    @staticmethod
    def _normalize_unit(unit) -> str:
        """Normalize unit of measurement."""
        if not unit or pd.isna(unit):
            return "Each"
        
        unit_str = str(unit).strip().lower()
        if re.match(r'^\d*\.?\d*$', unit_str.replace('.', '')):
            return "Each"
        
        unit_map = {
            'sqm': 'Sqm', 'sq.m': 'Sqm', 'sq m': 'Sqm', 'square meter': 'Sqm', 'm2': 'Sqm',
            'cum': 'Cum', 'cu.m': 'Cum', 'cu m': 'Cum', 'cubic meter': 'Cum', 'm3': 'Cum',
            'mtr': 'Mtr', 'm': 'Mtr', 'meter': 'Mtr', 'metre': 'Mtr', 'rm': 'Rmt',
            'kg': 'Kg', 'kilogram': 'Kg', 'kgs': 'Kg',
            'nos': 'Nos', 'no': 'Nos', 'number': 'Nos', 'each': 'Each', 'pcs': 'Nos',
            'litre': 'Ltr', 'ltr': 'Ltr', 'l': 'Ltr', 'liter': 'Ltr',
            'ton': 'Ton', 'tonne': 'Ton', 'mt': 'Ton',
            'rmt': 'Rmt', 'running meter': 'Rmt', 'running m': 'Rmt'
        }
        return unit_map.get(unit_str, unit_str.title())