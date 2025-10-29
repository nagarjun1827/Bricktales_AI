import pandas as pd
import json
import re
from typing import Dict, List, Any
from .base_agent import BaseAgent

class SheetStructureAnalyzerAgent(BaseAgent):
    def execute(self, df: pd.DataFrame) -> Dict[str, Any]:
        print(f"      Analyzing sheet structure (shape: {df.shape})...")
        
        sample_rows = min(30, len(df))
        sheet_text = []
        for idx in range(sample_rows):
            row_vals = [str(v) for v in df.iloc[idx].values if pd.notna(v)]
            if row_vals:
                sheet_text.append(f"Row {idx}: {' | '.join(row_vals[:10])}")
        
        analysis_text = '\n'.join(sheet_text)

        prompt = f"""You are a BOQ (Bill of Quantities) sheet structure analyzer. Analyze this Excel sheet data.

Sheet Data:
{analysis_text}

Determine:
1. **has_header**: Does this sheet have a header row with column names? (true/false)
2. **header_row**: If has_header is true, which row number is the header? (0-based index, or -1 if no header)
3. **data_start_row**: Which row does the actual data start? (0-based index)
4. **column_structure**: What type of columns exist? Describe each column position.

Return a JSON object:
{{
    "has_header": true/false,
    "header_row": number (-1 if no header),
    "data_start_row": number,
    "column_structure": [
        {{"position": 0, "type": "item_code", "description": "Item numbers"}},
        {{"position": 1, "type": "description", "description": "Work description"}}
    ]
}}

Return ONLY valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            result_text = self._clean_json_response(response.text)
            structure = json.loads(result_text)
            
            print(f"      Has header: {structure['has_header']}")
            print(f"      Header row: {structure['header_row']}")
            print(f"      Data starts at row: {structure['data_start_row']}")
            
            return structure
            
        except Exception as e:
            print(f"      Structure analysis failed: {e}")
            return self._fallback_analysis(df)
    
    def _fallback_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        first_row = df.iloc[0]
        first_row_text = ' '.join([str(v).lower() for v in first_row.values if pd.notna(v)])
        
        header_keywords = ['item', 'description', 'quantity', 'unit', 'rate', 'amount', 'code', 'no']
        has_header_keyword = any(kw in first_row_text for kw in header_keywords)
        
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
        columns = []
        type_map = {
            'item': 'item_code', 'code': 'item_code', 'no': 'item_code', 'sl': 'item_code',
            'description': 'description', 'particular': 'description', 'work': 'description', 'desc': 'description',
            'unit': 'unit', 'uom': 'unit', 'measurement': 'unit',
            'quantity': 'quantity', 'qty': 'quantity', 'qnty': 'quantity',
            'rate': 'rate', 'price': 'rate', 'unit rate': 'rate',
            'amount': 'amount', 'total': 'amount', 'value': 'amount', 'amt': 'amount'
        }
        
        for idx, col_name in enumerate(header_row.values):
            if pd.isna(col_name):
                continue
            
            col_str = str(col_name).lower().strip()
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
        columns = []
        
        for idx in range(len(df.columns)):
            sample_values = df.iloc[:10, idx].dropna().astype(str)
            
            if len(sample_values) == 0:
                continue
            
            is_numeric = all(re.match(r'^-?\d*\.?\d+$', str(v).replace(',', '')) for v in sample_values)
            is_short_text = all(len(str(v)) < 20 for v in sample_values)
            is_long_text = any(len(str(v)) > 50 for v in sample_values)
            
            if idx == 0 and is_numeric:
                col_type = 'item_code'
            elif is_long_text:
                col_type = 'description'
            elif is_short_text and not is_numeric:
                col_type = 'unit'
            elif is_numeric:
                col_type = 'quantity' if idx <= 3 else 'rate' if idx == 4 else 'amount'
            else:
                col_type = 'unknown'
            
            columns.append({
                'position': idx,
                'type': col_type,
                'description': f'Column {idx}'
            })
        
        return columns