"""
Service for processing store BOQ files (with rates and pricing) from URL.
"""
import json
import time
import requests
import io
import logging
from typing import Dict, Any
import pandas as pd
import google.generativeai as genai

from models.store_boq_models import ProjectInfo, StoreBOQProjectInfo, LocationInfo, BOQFileInfo
from repositories.store_boq import StoreBOQRepository
from agents.gemini_tools import (
    AnalyzeSheetStructureTool,
    ExtractProjectInfoTool,
    ExtractLocationInfoTool,
)
from agents.item_extractor import ItemExtractorAgent
from core.settings import settings
from datetime import datetime

logger = logging.getLogger(__name__)


class StoreBOQProcessor:
    """Processes store BOQ files from URL with automatic embedding generation."""
    
    def __init__(self):
        self.repo = StoreBOQRepository()
        self.item_extractor = ItemExtractorAgent()
        self.project_tool = ExtractProjectInfoTool()
        self.location_tool = ExtractLocationInfoTool()
        self.structure_tool = AnalyzeSheetStructureTool()
        
        # Configure Gemini for embeddings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "models/text-embedding-004"

    def process_file_from_url(
        self, 
        file_url: str, 
        uploaded_by: str = "user"
    ) -> Dict[str, Any]:
        """
        Process store BOQ file from URL with automatic embedding generation.
        
        Args:
            file_url: URL to Excel file
            uploaded_by: User identifier
            
        Returns:
            Processing results with embeddings
        """
        logger.info(f"Storage file URL: {file_url}")

        start_time = time.time()

        try:
            # Step 1: Download and read Excel file from URL
            logger.info("Step 1: Downloading and reading Excel file from URL")
            sheets = self._read_excel_from_url(file_url)
            logger.info(f"Read {len(sheets)} sheets")

            # Step 2-6: Extract and save project/store_project/location
            first_sheet = list(sheets.values())[0]
            project_id, store_project_id, location_id = self._process_metadata(
                first_sheet, uploaded_by
            )

            # Step 7: Create BOQ file record
            logger.info("Step 7: Creating BOQ file record")
            boq_id = self._create_boq_file(store_project_id, file_url, uploaded_by)
            logger.info(f"BOQ ID: {boq_id}")

            # Step 8: Process sheets and extract items
            logger.info("Step 8: Processing sheets and extracting items")
            boq_sheets = self._filter_sheets(sheets)
            all_items = self._extract_all_items(boq_sheets, boq_id, location_id)

            # Step 9: Insert items
            logger.info(f"Step 9: Inserting {len(all_items)} items")
            self.repo.insert_boq_items_batch(all_items)
            logger.info("Items inserted successfully")

            # Step 10: Get totals
            logger.info("Step 10: Calculating totals")
            totals = self.repo.get_boq_totals(boq_id, all_items)

            # Step 11: Generate embeddings
            embedding_result = self._generate_embeddings(boq_id, all_items)

            elapsed = time.time() - start_time

            self._log_summary(
                project_id, store_project_id, location_id, boq_id, 
                totals, embedding_result, elapsed
            )

            return {
                "success": True,
                "project_id": project_id,
                "store_project_id": store_project_id,
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
            logger.error(f"Processing failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _read_excel_from_url(self, file_url: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file at URL."""
        try:
            logger.info("Fetching file from URL")
            response = requests.get(file_url, timeout=120)
            response.raise_for_status()
            
            logger.info(f"File downloaded ({len(response.content)} bytes)")
            logger.info("Reading Excel sheets")
            
            # Read Excel from bytes
            excel_file = pd.ExcelFile(io.BytesIO(response.content))
            sheets = {
                name: pd.read_excel(io.BytesIO(response.content), sheet_name=name) 
                for name in excel_file.sheet_names
            }
            
            return sheets
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to read Excel file: {str(e)}")

    def _process_metadata(self, first_sheet: pd.DataFrame, uploaded_by: str) -> tuple:
        """Extract and save project, store project, and location metadata."""
        logger.info("Step 2: Extracting project information")
        text_sample = self._extract_text(first_sheet, 50)
        
        # Extract project
        project_json = self.project_tool._run(text_sample)
        project_data = json.loads(project_json)
        project_info = self._build_project_info(project_data, uploaded_by)
        
        logger.info(f"Project: {project_info.project_name}")
        
        # Insert project
        logger.info("Step 3: Saving project")
        project_id = self.repo.insert_project(project_info)
        logger.info(f"Project ID: {project_id}")
        
        # Create store BOQ project
        logger.info("Step 4: Creating store BOQ project")
        store_project_info = self._build_store_project_info(
            project_id, project_data, uploaded_by
        )
        store_project_id = self.repo.insert_store_boq_project(store_project_info)
        logger.info(f"Store Project ID: {store_project_id}")
        
        # Extract location
        logger.info("Step 5: Extracting location")
        location_json = self.location_tool._run(text_sample)
        location_data = json.loads(location_json)
        location_info = self._build_location_info(location_data, store_project_id)
        
        logger.info(f"Location: {location_info.location_name}")
        
        # Insert location
        logger.info("Step 6: Saving location")
        location_id = self.repo.insert_location(location_info)
        logger.info(f"Location ID: {location_id}")
        
        return project_id, store_project_id, location_id

    def _create_boq_file(self, store_project_id: int, file_url: str, uploaded_by: str) -> int:
        """Create BOQ file record."""
        # Extract filename from URL or use generic name
        try:
            from urllib.parse import urlparse
            parsed = urlparse(file_url)
            filename = parsed.path.split('/')[-1] or "boq_from_url.xlsx"
        except:
            filename = "boq_from_url.xlsx"
            
        file_info = BOQFileInfo(
            store_project_id=store_project_id,
            file_name=filename,
            file_path=file_url,  # Store URL instead of local path
            created_by=uploaded_by,
        )
        return self.repo.insert_boq_file(file_info)

    def _extract_all_items(self, sheets: Dict, boq_id: int, location_id: int):
        """Extract items from all sheets."""
        all_items = []
        logger.info(f"Processing {len(sheets)} sheet(s)")
        
        for sheet_name, sheet_df in sheets.items():
            logger.info(f"Processing sheet: {sheet_name}")
            items = self._process_sheet(sheet_df, sheet_name, boq_id, location_id)
            all_items.extend(items)
            
        return all_items

    def _process_sheet(self, sheet_df: pd.DataFrame, sheet_name: str, boq_id: int, location_id: int):
        """Process single sheet."""
        sheet_text = self._sheet_to_text(sheet_df, 30)
        structure_json = self.structure_tool._run(sheet_text, sheet_name)
        structure = json.loads(structure_json)
        
        logger.debug("Structure analyzed")
        items = self.item_extractor.execute(sheet_df, structure, boq_id, location_id)
        return items

    def _generate_embeddings(self, boq_id: int, items: list) -> Dict:
        """Generate embeddings for all items."""
        if not items:
            return {"generated": False, "count": 0, "time": 0}

        logger.info("Embedding Generation Started")
        
        embedding_start = time.time()
        
        try:
            # Initialize pgvector
            logger.info("Initializing vector database")
            self.repo.ensure_vector_extension()
            logger.info("Database ready")
            
            # Get items without embeddings
            items_to_embed = self.repo.get_items_without_embeddings(boq_id)
            
            if not items_to_embed:
                logger.info("All items already have embeddings")
                return {"generated": False, "count": 0, "time": 0}
            
            logger.info(f"Generating embeddings for {len(items_to_embed)} items")
            
            # Generate embeddings in batches
            batch_size = 100
            total_generated = 0
            
            for i in range(0, len(items_to_embed), batch_size):
                batch = items_to_embed[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(items_to_embed) + batch_size - 1) // batch_size
                
                logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} items)")
                
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
                        logger.warning(f"Failed for item {item.item_id}: {e}")
                        continue
                
                # Save to database
                if item_ids and embeddings:
                    updated = self.repo.update_embeddings_batch(item_ids, embeddings)
                    total_generated += updated
                    logger.info(f"Saved {updated} embeddings")
            
            embedding_time = time.time() - embedding_start
            
            logger.info("Embeddings complete")
            logger.info(f"Total generated: {total_generated}")
            logger.info(f"Time: {embedding_time:.2f}s")
            
            return {
                "generated": True,
                "count": total_generated,
                "time": embedding_time
            }
            
        except Exception as e:
            logger.error(f"Embedding error: {e}", exc_info=True)
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
    def _build_project_info(data: dict, uploaded_by: str) -> ProjectInfo:
        """Build ProjectInfo from extracted data."""
        start_date = f"{data['start_year']}-01-01" if data.get('start_year') else None
        end_date = f"{data['end_year']}-12-31" if data.get('end_year') else (
            f"{data['start_year']}-12-31" if data.get('start_year') else None
        )
        
        return ProjectInfo(
            project_name=data.get("project_name", "BOQ Project"),
            project_code=data.get("project_code", f"PROJ-{datetime.now():%Y%m%d}"),
            project_type="boq",
            client_name=data.get("client_name"),
            start_date=start_date,
            end_date=end_date,
            created_by=uploaded_by,
        )

    @staticmethod
    def _build_store_project_info(
        project_id: int, 
        data: dict, 
        uploaded_by: str
    ) -> StoreBOQProjectInfo:
        """Build StoreBOQProjectInfo from extracted data."""
        project_name = data.get("project_name", "BOQ Project")
        return StoreBOQProjectInfo(
            project_id=project_id,
            store_project_name=f"{project_name} - Store",
            store_project_code=f"STORE-{datetime.now():%Y%m%d%H%M}",
            created_by=uploaded_by,
        )

    @staticmethod
    def _build_location_info(data: dict, store_project_id: int) -> LocationInfo:
        """Build LocationInfo from extracted data."""
        address_parts = [data.get("location_name", "Unknown")]
        if data.get("city"):
            address_parts.append(data["city"])
        if data.get("state"):
            address_parts.append(data["state"])
            
        return LocationInfo(
            store_project_id=store_project_id,
            location_name=data.get("location_name", "Unknown"),
            address=", ".join(address_parts),
        )

    @staticmethod
    def _log_summary(
        project_id, store_project_id, location_id, boq_id, 
        totals, embedding_result, elapsed
    ):
        """Log processing summary."""
        logger.info("Processing Summary Generating")

        logger.info(f"Project ID: {project_id}")
        logger.info(f"Store Project ID: {store_project_id}")
        logger.info(f"Location ID: {location_id}")
        logger.info(f"BOQ File ID: {boq_id}")
        logger.info(f"Total Items: {totals['item_count']}")
        logger.info(f"Supply Amount: ₹{totals['total_supply']:,.2f}")
        logger.info(f"Labour Amount: ₹{totals['total_labour']:,.2f}")
        logger.info(f"Total Amount: ₹{totals['total_amount']:,.2f}")
        if embedding_result["generated"]:
            logger.info(f"Embeddings Generated: {embedding_result['count']}")
            logger.info(f"Embedding Time: {embedding_result['time']:.2f}s")
        logger.info(f"Total Time: {elapsed:.2f}s")