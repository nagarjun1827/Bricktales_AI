import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from app.models.domain import ProjectInfo, LocationInfo, BOQFileInfo, BOQItem
from app.repositories.boq_repository import BoQRepository
from app.agents.langchain_tools import (
    AnalyzeSheetStructureTool,
    ExtractProjectInfoTool,
    ExtractLocationInfoTool,
)
from datetime import datetime
import re

class SmartItemExtractor:
    def extract_items(
        self, df: pd.DataFrame, structure: Dict[str, Any], boq_id: int, location_id: int
    ) -> List[BOQItem]:
        data_start = structure["data_start_row"]
        col_map = {col["type"]: col["position"] for col in structure["column_structure"]}

        print(f"      ✓ Column mapping: {col_map}")

        items = []
        data_df = df.iloc[data_start:]

        for _, row in data_df.iterrows():
            if row.isna().all():
                continue

            item_code = self._get_value(row, col_map.get("item_code"))
            description = self._get_value(row, col_map.get("description"), default="")
            unit = self._get_value(row, col_map.get("unit"), default="Each")
            quantity = self._extract_numeric(self._get_value(row, col_map.get("quantity")))
            rate = self._extract_numeric(self._get_value(row, col_map.get("rate")))
            amount = self._extract_numeric(self._get_value(row, col_map.get("amount")))

            if not description or (quantity == 0 and rate == 0 and amount == 0):
                continue

            if len(str(description).strip()) < 5:
                continue

            quantity = quantity if quantity > 0 else (amount / rate if rate > 0 else 0)

            item = BOQItem(
                boq_id=boq_id,
                item_code=str(item_code) if item_code is not None else None,
                item_description=str(description).strip(),
                unit_of_measurement=self._normalize_unit(unit),
                quantity=quantity,
                supply_unit_rate=rate,
                location_id=location_id,
                created_by="system",
            )

            item.calculate_amounts()
            items.append(item)

        print(f"      ✓ Extracted {len(items)} valid items")
        return items

    def _get_value(self, row, position, default=None):
        if position is None or position >= len(row):
            return default
        val = row.iloc[position]
        return val if pd.notna(val) else default

    def _extract_numeric(self, value) -> float:
        if value is None or pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).replace(",", "")
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value_str)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return 0.0
        return 0.0

    def _normalize_unit(self, unit) -> str:
        if not unit or pd.isna(unit):
            return "Each"

        unit_str = str(unit).strip()
        if unit_str.replace(".", "").replace("-", "").isdigit():
            return "Each"

        unit_lower = unit_str.lower()
        unit_map = {
            "sqm": "Sqm",
            "sq.m": "Sqm",
            "sq m": "Sqm",
            "cum": "Cum",
            "cu.m": "Cum",
            "cu m": "Cum",
            "mtr": "Mtr",
            "m": "Mtr",
            "meter": "Mtr",
            "kg": "Kg",
            "nos": "Nos",
            "no": "Nos",
            "litre": "Ltr",
            "ltr": "Ltr",
            "ton": "Ton",
            "rm": "Rmt",
            "rmt": "Rmt",
        }
        return unit_map.get(unit_lower, unit_str)

class LangChainBOQService:
    def __init__(self):
        self.repo = BoQRepository()
        self.item_extractor = SmartItemExtractor()
        self.project_tool = ExtractProjectInfoTool()
        self.location_tool = ExtractLocationInfoTool()
        self.structure_tool = AnalyzeSheetStructureTool()

    def process_file(self, file_path: str, uploaded_by: str = "system") -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"🤖 LANGCHAIN MULTI-AGENT BOQ PROCESSOR")
        print(f"{'='*70}")
        print(f"File: {Path(file_path).name}\n")

        start_time = time.time()

        try:
            # Step 1: Read file
            print("🔍 AGENT: File Reader")
            excel_file = pd.ExcelFile(file_path)
            sheets = {name: pd.read_excel(file_path, sheet_name=name) for name in excel_file.sheet_names}
            print(f"   ✓ Read {len(sheets)} sheets: {', '.join(sheets.keys())}\n")

            # Step 2: Extract project
            print("🔍 AGENT: Project Extractor")
            first_sheet = list(sheets.values())[0]
            text_sample = self._extract_text(first_sheet, 50)

            project_json = self.project_tool._run(text_sample)
            project_data = json.loads(project_json)

            start_date = None
            end_date = None
            if project_data.get("start_year"):
                start_date = f"{project_data['start_year']}-01-01"
            if project_data.get("end_year"):
                end_date = f"{project_data['end_year']}-12-31"
            elif project_data.get("start_year"):
                end_date = f"{project_data['start_year']}-12-31"

            project_info = ProjectInfo(
                project_name=project_data.get("project_name", "BOQ Project"),
                project_code=project_data.get(
                    "project_code", f"PROJ-{datetime.now():%Y%m%d}"
                ),
                client_name=project_data.get("client_name"),
                start_date=start_date,
                end_date=end_date,
            )

            print(f"   ✓ Project: {project_info.project_name}")
            print(f"   ✓ Code: {project_info.project_code}\n")

            # Step 3: Insert project
            print("💾 DATABASE: Inserting project")
            project_id = self.repo.insert_project(project_info)
            print(f"   ✓ Project ID: {project_id}\n")

            # Step 4: Extract location
            print("🔍 AGENT: Location Extractor")
            location_json = self.location_tool._run(text_sample)
            location_data = json.loads(location_json)

            address_parts = [location_data.get("location_name", "Unknown")]
            if location_data.get("city"):
                address_parts.append(location_data["city"])
            if location_data.get("state"):
                address_parts.append(location_data["state"])

            location_info = LocationInfo(
                project_id=project_id,
                location_name=location_data.get("location_name", "Unknown"),
                address=", ".join(address_parts),
            )

            print(f"   ✓ Location: {location_info.location_name}\n")

            # Step 5: Insert location
            print("💾 DATABASE: Inserting location")
            location_id = self.repo.insert_location(location_info)
            print(f"   ✓ Location ID: {location_id}\n")

            # Step 6: Insert file metadata
            print("💾 DATABASE: Inserting file metadata")
            file_info = BOQFileInfo(
                project_id=project_id,
                file_name=Path(file_path).name,
                file_path=file_path,
                created_by=uploaded_by,
            )
            boq_id = self.repo.insert_boq_file(file_info)
            print(f"   ✓ BOQ File ID: {boq_id}\n")

            # Step 7: Filter sheets
            print("🔍 AGENT: Sheet Filter")
            boq_sheets = self._filter_sheets(sheets)
            print(f"   ✓ Processing {len(boq_sheets)} BOQ sheet(s)\n")

            # Step 8: Process sheets
            all_items = []
            for sheet_name, sheet_df in boq_sheets.items():
                print(f"   📄 Processing: {sheet_name}")

                # Analyze structure
                sheet_text = self._sheet_to_text(sheet_df, 30)
                structure_json = self.structure_tool._run(sheet_text, sheet_name)
                structure = json.loads(structure_json)

                print(f"      🔍 Analyzing sheet structure (shape: {sheet_df.shape})...")
                print(f"      ✓ Has header: {structure['has_header']}")
                print(f"      ✓ Header row: {structure['header_row']}")
                print(f"      ✓ Data starts at row: {structure['data_start_row']}")
                print(f"      ✓ Identified {len(structure['column_structure'])} columns")

                # Extract items
                print("      🔍 Extracting items with intelligent parsing...")
                items = self.item_extractor.extract_items(sheet_df, structure, boq_id, location_id)
                all_items.extend(items)
                print()

            # Step 9: Insert items
            print("💾 DATABASE: Inserting BOQ items")
            self.repo.insert_boq_items_batch(all_items)
            print(f"   ✓ Inserted {len(all_items)} items\n")

            # Step 10: Get totals
            print("📊 Fetching calculated totals from database...")
            totals = self.repo.get_boq_totals(boq_id)

            elapsed = time.time() - start_time

            print(f"{'='*70}")
            print("✓ PROCESSING COMPLETE")
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
                "success": True,
                "project_id": project_id,
                "boq_id": boq_id,
                "total_items": totals["item_count"],
                "total_supply": totals["total_supply"],
                "total_labour": totals["total_labour"],
                "total_amount": totals["total_amount"],
                "processing_time": elapsed,
            }

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _extract_text(self, df: pd.DataFrame, max_rows: int) -> str:
        lines = []
        for _, row in df.head(max_rows).iterrows():
            text = " ".join([str(v) for v in row.values if pd.notna(v)])
            if text.strip():
                lines.append(text)
        return "\n".join(lines)[:5000]

    def _sheet_to_text(self, df: pd.DataFrame, max_rows: int) -> str:
        lines = []
        for idx in range(min(max_rows, len(df))):
            row_vals = [str(v) for v in df.iloc[idx].values if pd.notna(v)]
            if row_vals:
                lines.append(f"Row {idx}: {' | '.join(row_vals[:10])}")
        return "\n".join(lines)

    def _filter_sheets(self, sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        skip = ["summary", "assumption", "note", "index", "cover", "terms"]
        boq_sheets = {}
        for name, df in sheets.items():
            if any(kw in name.lower() for kw in skip):
                continue
            if len(df) < 5 or len(df.columns) < 3:
                continue
            boq_sheets[name] = df
        return boq_sheets