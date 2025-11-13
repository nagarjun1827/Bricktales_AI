"""
Item extraction agent for BOQ processing.
"""
import logging
import pandas as pd
import re
from typing import List, Optional, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Define BOQItem as a Pydantic model (used for data transfer)
class BOQItem(BaseModel):
    boq_id: int
    item_code: Optional[str] = None
    item_description: str
    unit_of_measurement: str
    quantity: float
    supply_unit_rate: float = 0.0
    labour_unit_rate: float = 0.0
    location_id: Optional[int] = None
    created_by: str = "system"
    
    @property
    def supply_amount(self) -> float:
        return self.quantity * self.supply_unit_rate
    
    @property
    def labour_amount(self) -> float:
        return self.quantity * self.labour_unit_rate
    
    @property
    def total_amount(self) -> float:
        return self.supply_amount + self.labour_amount


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
        
        # Track statistics
        items_with_supply_rate = 0
        items_with_labour_rate = 0
        
        for idx, row in data_df.iterrows():
            if row.isna().all():
                continue
            
            # Extract values with proper None handling
            item_code = self._get_value(row, col_map.get('item_code'))
            description = self._get_value(row, col_map.get('description'), default='')
            unit = self._get_value(row, col_map.get('unit'), default='Each')
            quantity = self._extract_numeric(self._get_value(row, col_map.get('quantity')))
            
            # Extract supply rate and amount
            supply_rate = self._extract_numeric(self._get_value(row, col_map.get('supply_rate')))
            supply_amount = self._extract_numeric(self._get_value(row, col_map.get('supply_amount')))
            
            # Extract labour rate and amount
            labour_rate = self._extract_numeric(self._get_value(row, col_map.get('labour_rate')))
            labour_amount = self._extract_numeric(self._get_value(row, col_map.get('labour_amount')))
            
            # Extract total amount
            total_amount = self._extract_numeric(self._get_value(row, col_map.get('total_amount')))
            
            # Skip invalid rows
            if not description:
                continue
            if len(str(description).strip()) < 5:
                continue
            if quantity == 0 and supply_rate == 0 and labour_rate == 0 and supply_amount == 0 and labour_amount == 0 and total_amount == 0:
                continue
            
            # Derive missing values intelligently
            
            # 1. Derive quantity if missing
            if quantity == 0:
                if supply_rate > 0 and supply_amount > 0:
                    quantity = round(supply_amount / supply_rate, 4)
                elif labour_rate > 0 and labour_amount > 0:
                    quantity = round(labour_amount / labour_rate, 4)
                elif (supply_rate + labour_rate) > 0 and total_amount > 0:
                    quantity = round(total_amount / (supply_rate + labour_rate), 4)
            
            # 2. Derive supply_rate if missing but have supply_amount and quantity
            if supply_rate == 0 and supply_amount > 0 and quantity > 0:
                supply_rate = round(supply_amount / quantity, 2)
            
            # 3. Derive labour_rate if missing but have labour_amount and quantity
            if labour_rate == 0 and labour_amount > 0 and quantity > 0:
                labour_rate = round(labour_amount / quantity, 2)
            
            # 4. If we have total_amount but no supply/labour breakdown, try to derive
            if total_amount > 0 and quantity > 0:
                if supply_rate == 0 and labour_rate == 0:
                    # Assume all is supply if no labour info
                    if labour_amount == 0:
                        supply_rate = round(total_amount / quantity, 2)
                    # If labour_amount exists, derive labour_rate
                    elif labour_amount > 0:
                        labour_rate = round(labour_amount / quantity, 2)
                        remaining = total_amount - labour_amount
                        if remaining > 0:
                            supply_rate = round(remaining / quantity, 2)
            
            # Final validation - skip if no meaningful data
            if quantity == 0:
                continue
            
            # Create item
            item = BOQItem(
                boq_id=boq_id,
                item_code=str(item_code) if item_code is not None else None,
                item_description=str(description).strip(),
                unit_of_measurement=self._normalize_unit(unit),
                quantity=quantity,
                supply_unit_rate=supply_rate,
                labour_unit_rate=labour_rate,
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
    
    def _find_rate_column(self, df: pd.DataFrame, rate_type: str) -> Optional[int]:
        """
        Fallback method to find rate columns using pattern matching.
        
        Args:
            df: DataFrame to search
            rate_type: 'supply' or 'labour'
        
        Returns:
            Column position or None
        """
        if rate_type == 'supply':
            patterns = [
                r'supply.*rate', r'material.*rate', r'supply.*unit',
                r'^rate$', r'unit.*rate', r'rate.*unit', r'mat.*rate'
            ]
        else:  # labour
            patterns = [
                r'labour.*rate', r'labor.*rate', r'labour.*unit',
                r'labor.*unit', r'lab.*rate'
            ]
        
        # Search in column headers
        for idx, col in enumerate(df.columns):
            col_str = str(col).lower().strip()
            for pattern in patterns:
                if re.search(pattern, col_str, re.IGNORECASE):
                    # Additional check: verify column has numeric data
                    if self._is_numeric_column(df.iloc[:, idx]):
                        return idx
        
        return None
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column contains mostly numeric data."""
        # Take first 20 non-null values
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return False
        
        numeric_count = 0
        for val in sample:
            if self._extract_numeric(val) > 0:
                numeric_count += 1
        
        # Consider numeric if >50% of samples are numeric
        return numeric_count / len(sample) > 0.5
    
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