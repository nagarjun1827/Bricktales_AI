"""
Service for processing store BOQ files (with rates and pricing).
"""
import json
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import google.generativeai as genai

from models.domain import ProjectInfo, LocationInfo, BOQFileInfo
from repositories.store_boq_repository import StoreBOQRepository
from agents.gemini_tools import (
    AnalyzeSheetStructureTool,
    ExtractProjectInfoTool,
    ExtractLocationInfoTool,
)
from agents.item_extractor import ItemExtractorAgent
from core.settings import settings
from datetime import datetime


class StoreBOQProcessor:
    """Processes store BOQ files with automatic embedding generation."""
    
    def __init__(self):
        self.repo = StoreBOQRepository()
        self.item_extractor = ItemExtractorAgent()
        self.project_tool = ExtractProjectInfoTool()
        self.location_tool = ExtractLocationInfoTool()
        self.structure_tool = AnalyzeSheetStructureTool()
        
        # Configure Gemini for embeddings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "models/text-embedding-004"

    def process_file(
        self, 
        file_path: str, 
        uploaded_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Process store BOQ file with automatic embedding generation.
        
        Args:
            file_path: Path to Excel file
            uploaded_by: User identifier
            
        Returns:
            Processing results with embeddings
        """
        print(f"\n{'='*70}")
        print(f"🤖 STORE BOQ PROCESSOR")
        print(f"🧠 WITH AUTOMATIC EMBEDDING GENERATION")
        print(f"{'='*70}")
        print(f"File: {Path(file_path).name}\n")

        start_time = time.time()

        try:
            # Step 1: Read Excel file
            print("📖 Step 1: Reading Excel file...")
            sheets = self._read_excel_file(file_path)
            print(f"   ✓ Read {len(sheets)} sheets\n")

            # Step 2-5: Extract and save project/location
            first_sheet = list(sheets.values())[0]
            project_id, location_id = self._process_metadata(first_sheet, uploaded_by)

            # Step 6: Create BOQ file record
            print("💾 Step 6: Creating BOQ file record...")
            boq_id = self._create_boq_file(project_id, file_path, uploaded_by)
            print(f"   ✓ BOQ ID: {boq_id}\n")

            # Step 7: Process sheets and extract items
            print("📊 Step 7: Processing sheets and extracting items...")
            boq_sheets = self._filter_sheets(sheets)
            all_items = self._extract_all_items(boq_sheets, boq_id, location_id)

            # Step 8: Insert items
            print(f"\n💾 Step 8: Inserting {len(all_items)} items...")
            self.repo.insert_boq_items_batch(all_items)
            print(f"   ✓ Items inserted\n")

            # Step 9: Get totals
            print("📊 Step 9: Calculating totals...")
            totals = self.repo.get_boq_totals(boq_id)

            # Step 10: Generate embeddings
            embedding_result = self._generate_embeddings(boq_id, all_items)

            elapsed = time.time() - start_time

            self._print_summary(
                project_id, location_id, boq_id, 
                totals, embedding_result, elapsed
            )

            return {
                "success": True,
                "project_id": project_id,
                "boq_id": boq_id,
                "total_items": totals["item_count"],
                "total_supply": totals["total_supply"],
                "total_labour": totals["total_labour"],
                "total_amount": totals["total_amount"],
                "embeddings_generated": embedding_result["generated"],
                "embeddings_count": embedding_result["count"],
                "embedding_time": embedding_result["time"],
                "processing_time": elapsed,
            }

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _read_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file."""
        excel_file = pd.ExcelFile(file_path)
        return {
            name: pd.read_excel(file_path, sheet_name=name) 
            for name in excel_file.sheet_names
        }

    def _process_metadata(self, first_sheet: pd.DataFrame, uploaded_by: str) -> tuple:
        """Extract and save project and location metadata."""
        print("🔍 Step 2: Extracting project information...")
        text_sample = self._extract_text(first_sheet, 50)
        
        # Extract project
        project_json = self.project_tool._run(text_sample)
        project_data = json.loads(project_json)
        project_info = self._build_project_info(project_data)
        
        print(f"   ✓ Project: {project_info.project_name}")
        
        # Insert project
        print("\n💾 Step 3: Saving project...")
        project_id = self.repo.insert_project(project_info)
        print(f"   ✓ Project ID: {project_id}")
        
        # Extract location
        print("\n🔍 Step 4: Extracting location...")
        location_json = self.location_tool._run(text_sample)
        location_data = json.loads(location_json)
        location_info = self._build_location_info(location_data, project_id)
        
        print(f"   ✓ Location: {location_info.location_name}")
        
        # Insert location
        print("\n💾 Step 5: Saving location...")
        location_id = self.repo.insert_location(location_info)
        print(f"   ✓ Location ID: {location_id}\n")
        
        return project_id, location_id

    def _create_boq_file(self, project_id: int, file_path: str, uploaded_by: str) -> int:
        """Create BOQ file record."""
        file_info = BOQFileInfo(
            project_id=project_id,
            file_name=Path(file_path).name,
            file_path=file_path,
            created_by=uploaded_by,
        )
        return self.repo.insert_boq_file(file_info)

    def _extract_all_items(self, sheets: Dict, boq_id: int, location_id: int):
        """Extract items from all sheets."""
        all_items = []
        print(f"   Processing {len(sheets)} sheet(s)...\n")
        
        for sheet_name, sheet_df in sheets.items():
            print(f"   📄 {sheet_name}")
            items = self._process_sheet(sheet_df, sheet_name, boq_id, location_id)
            all_items.extend(items)
            
        return all_items

    def _process_sheet(self, sheet_df: pd.DataFrame, sheet_name: str, boq_id: int, location_id: int):
        """Process single sheet."""
        sheet_text = self._sheet_to_text(sheet_df, 30)
        structure_json = self.structure_tool._run(sheet_text, sheet_name)
        structure = json.loads(structure_json)
        
        print(f"      ✓ Structure analyzed")
        items = self.item_extractor.execute(sheet_df, structure, boq_id, location_id)
        return items

    def _generate_embeddings(self, boq_id: int, items: list) -> Dict:
        """Generate embeddings for all items."""
        if not items:
            return {"generated": False, "count": 0, "time": 0}

        print(f"\n{'='*70}")
        print("🧠 AUTOMATIC EMBEDDING GENERATION")
        print(f"{'='*70}\n")
        
        embedding_start = time.time()
        
        try:
            # Initialize pgvector
            print("🔧 Initializing vector database...")
            self.repo.ensure_vector_extension()
            print("   ✓ Database ready\n")
            
            # Get items without embeddings
            items_to_embed = self.repo.get_items_without_embeddings(boq_id)
            
            if not items_to_embed:
                print("   ✓ All items already have embeddings\n")
                return {"generated": False, "count": 0, "time": 0}
            
            print(f"📝 Generating embeddings for {len(items_to_embed)} items...")
            
            # Generate embeddings in batches
            batch_size = 100
            total_generated = 0
            
            for i in range(0, len(items_to_embed), batch_size):
                batch = items_to_embed[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(items_to_embed) + batch_size - 1) // batch_size
                
                print(f"   Batch {batch_num}/{total_batches} ({len(batch)} items)...")
                
                # Generate embeddings
                item_ids = []
                embeddings = []
                
                for item in batch:
                    try:
                        result = genai.embed_content(
                            model=self.embedding_model,
                            content=item.item_description,
                            task_type="retrieval_document"
                        )
                        item_ids.append(item.item_id)
                        embeddings.append(result['embedding'])
                    except Exception as e:
                        print(f"      ⚠ Failed for item {item.item_id}: {e}")
                        continue
                
                # Save to database
                if item_ids and embeddings:
                    updated = self.repo.update_embeddings_batch(item_ids, embeddings)
                    total_generated += updated
                    print(f"   ✓ Saved {updated} embeddings")
            
            embedding_time = time.time() - embedding_start
            
            print(f"\n✓ Embeddings complete!")
            print(f"   Total generated: {total_generated}")
            print(f"   Time: {embedding_time:.2f}s\n")
            
            return {
                "generated": True,
                "count": total_generated,
                "time": embedding_time
            }
            
        except Exception as e:
            print(f"\n⚠ Embedding error: {e}\n")
            return {
                "generated": False,
                "count": 0,
                "time": time.time() - embedding_start
            }

    @staticmethod
    def _extract_text(df: pd.DataFrame, max_rows: int) -> str:
        """Extract text from dataframe."""
        lines = []
        for _, row in df.head(max_rows).iterrows():
            text = " ".join([str(v) for v in row.values if pd.notna(v)])
            if text.strip():
                lines.append(text)
        return "\n".join(lines)[:5000]

    @staticmethod
    def _sheet_to_text(df: pd.DataFrame, max_rows: int) -> str:
        """Convert sheet to text."""
        lines = []
        for idx in range(min(max_rows, len(df))):
            row_vals = [str(v) for v in df.iloc[idx].values if pd.notna(v)]
            if row_vals:
                lines.append(f"Row {idx}: {' | '.join(row_vals[:10])}")
        return "\n".join(lines)

    @staticmethod
    def _filter_sheets(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Filter out non-BOQ sheets."""
        skip = ["summary", "assumption", "note", "index", "cover", "terms"]
        return {
            name: df for name, df in sheets.items()
            if not any(kw in name.lower() for kw in skip)
            and len(df) >= 5 and len(df.columns) >= 3
        }

    @staticmethod
    def _build_project_info(data: dict) -> ProjectInfo:
        """Build ProjectInfo from extracted data."""
        start_date = f"{data['start_year']}-01-01" if data.get('start_year') else None
        end_date = f"{data['end_year']}-12-31" if data.get('end_year') else (
            f"{data['start_year']}-12-31" if data.get('start_year') else None
        )
        
        return ProjectInfo(
            project_name=data.get("project_name", "BOQ Project"),
            project_code=data.get("project_code", f"PROJ-{datetime.now():%Y%m%d}"),
            client_name=data.get("client_name"),
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _build_location_info(data: dict, project_id: int) -> LocationInfo:
        """Build LocationInfo from extracted data."""
        address_parts = [data.get("location_name", "Unknown")]
        if data.get("city"):
            address_parts.append(data["city"])
        if data.get("state"):
            address_parts.append(data["state"])
            
        return LocationInfo(
            project_id=project_id,
            location_name=data.get("location_name", "Unknown"),
            address=", ".join(address_parts),
        )

    @staticmethod
    def _print_summary(project_id, location_id, boq_id, totals, embedding_result, elapsed):
        """Print processing summary."""
        print(f"\n{'='*70}")
        print("✓ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Project ID:           {project_id}")
        print(f"Location ID:          {location_id}")
        print(f"BOQ File ID:          {boq_id}")
        print(f"Total Items:          {totals['item_count']}")
        print(f"Supply Amount:        ₹{totals['total_supply']:,.2f}")
        print(f"Labour Amount:        ₹{totals['total_labour']:,.2f}")
        print(f"Total Amount:         ₹{totals['total_amount']:,.2f}")
        if embedding_result["generated"]:
            print(f"Embeddings Generated: {embedding_result['count']}")
            print(f"Embedding Time:       {embedding_result['time']:.2f}s")
        print(f"Total Time:           {elapsed:.2f}s")
        print(f"{'='*70}\n")