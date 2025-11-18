import json
import time
import requests
import io
import logging
import random
import pandas as pd
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from models.store_boq_models import StoreBoqFile, StoreBoqLocation, StoreBoqProject
from models.project_models import Project
from repositories.store_boq import StoreBOQRepository
from agents.gemini_tools import AnalyzeSheetStructureTool, ExtractProjectInfoTool, ExtractLocationInfoTool
from agents.item_extractor import ItemExtractorAgent
from core.settings import settings

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

    async def process_file_from_url(self, file_url: str, uploaded_by: str = "user") -> Dict[str, Any]:
        """Process store BOQ file from URL with automatic embedding generation."""
        logger.info("QuoCO - Store BOQ Processing Started")
        logger.info(f"File URL: {file_url}")
        start_time = time.time()

        try:
            # Step 1: Download and read Excel file from URL
<<<<<<< HEAD
            sheets = self._read_excel_from_url(file_url)

            # Step 2: Extract project information
            first_sheet = list(sheets.values())[0]
            project_data = self._extract_project_info(first_sheet)
            logger.info(f"Project Name: {project_data['project_name']}")

            # Step 3: Extract location information
            location_data = self._extract_location_info(first_sheet)
            logger.info(f"Location: {location_data['location_name']}")

            # Step 4: Save project to database
            project_id = await self._save_project(project_data, uploaded_by)
            logger.info(f"Project ID: {project_id}")

            # Step 5: Save store project to database
            store_project_id = await self._save_store_project(project_id, project_data, uploaded_by)
            logger.info(f"Store Project ID: {store_project_id}")

            # Step 6: Save location to database
            location_id = await self._save_location(store_project_id, location_data)
            logger.info(f"Location ID: {location_id}")

            # Step 7: Extract file information and create BOQ file record
            file_info = self._extract_file_info(file_url)
            boq_id = await self._create_boq_file(store_project_id, file_info, uploaded_by)
            logger.info(f"BOQ ID: {boq_id}")

            # Step 8: Filter and process sheets
            boq_sheets = self._filter_sheets(sheets)
            logger.info(f"Found {len(boq_sheets)} BOQ sheets to process")
            for sheet_name in boq_sheets.keys():
                logger.info(f"  - {sheet_name}")

            # Step 9: Extract line items from all sheets
            all_items = self._extract_all_items(boq_sheets, boq_id, location_id)
            logger.info(f"Total items extracted: {len(all_items)}")

            # Step 10: Insert items into database
            await self.repo.insert_boq_items_batch(all_items)

            # Step 11: Calculate totals
            totals = await self.repo.get_boq_totals(boq_id, all_items)

            # Step 12: Generate embeddings
            embedding_result = await self._generate_embeddings(boq_id, all_items)
            elapsed = time.time() - start_time
           
            logger.info("PROCESSING COMPLETE - SUMMARY")
            logger.info(f"Total Items: {totals['item_count']}")
            logger.info(f"Supply Amount: ₹{totals['total_supply']:,.2f}")
            logger.info(f"Labour Amount: ₹{totals['total_labour']:,.2f}")
            logger.info(f"Total Amount: ₹{totals['total_amount']:,.2f}")
            logger.info(f"Total Processing Time: {elapsed:.2f}s")

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
            logger.error("Processing Failed")
            logger.error(f"Error: {e}", exc_info=True)
=======
            logger.error(f"Processing failed: {e}", exc_info=True)
>>>>>>> 3ecfa74882656e185b462f94606b26887e4e7cc3
            return {"success": False, "error": str(e)}

    def _read_excel_from_url(self, file_url: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file at URL."""
        try:
<<<<<<< HEAD
            logger.info("Fetching file from URL...")
            response = requests.get(file_url, timeout=120)
            response.raise_for_status()
            
            file_size = len(response.content)
            logger.info(f"File downloaded successfully ({file_size:,} bytes)")
            
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

<<<<<<< HEAD
    def _extract_project_info(self, first_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Extract project information from first sheet."""
        text_sample = self._extract_text(first_sheet, 50)
        try:
            project_json = self.project_tool.invoke({"text": text_sample})
            project_data = json.loads(project_json)
            return project_data
        except Exception as e:
            logger.warning(f"Project extraction failed: {e}")
            logger.info("Using default project information")
            return {
                "project_name": f"BOQ Project {datetime.now():%Y%m%d}",
                "project_code": f"PROJ-{datetime.now():%Y%m%d%H%M}",
                "client_name": None,
                "start_year": None,
                "end_year": None
            }

    def _extract_location_info(self, first_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Extract location information from first sheet."""
        text_sample = self._extract_text(first_sheet, 50)
        try:
            location_json = self.location_tool.invoke({"text": text_sample})
            location_data = json.loads(location_json)
            return location_data
        except Exception as e:
            logger.warning(f"Location extraction failed: {e}")
            logger.info("Using default location information")
            return {
                "location_name": "Unknown Location",
                "city": "Unknown City",
                "state": "Unknown State"
            }

    def _extract_file_info(self, file_url: str) -> Dict[str, str]:
        """Extract file information from URL."""
=======
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
>>>>>>> 3ecfa74882656e185b462f94606b26887e4e7cc3
        try:
            from urllib.parse import urlparse
            parsed = urlparse(file_url)
            filename = parsed.path.split('/')[-1] or "boq_name.xlsx"
            
            if filename.endswith('.xlsx'):
                file_type = 'xlsx'
            elif filename.endswith('.xls'):
                file_type = 'xls'
            else:
                file_type = 'xlsx'
            
            return {
                "file_name": filename,
                "file_path": file_url,
                "file_type": file_type
            }
        except Exception as e:
            logger.warning(f"Failed to parse filename from URL: {e}")
            return {
                "file_name": "boq_name.xlsx",
                "file_path": file_url,
                "file_type": "xlsx"
            }

    async def _save_project(self, project_data: dict, uploaded_by: str) -> int:
        """Save project to database."""
        project_info = self._build_project_info(project_data, uploaded_by)
        project_id = await self.repo.insert_project(project_info)
        return project_id

    async def _save_store_project(self, project_id: int, project_data: dict, uploaded_by: str) -> int:
        """Save store project to database."""
        store_project_info = self._build_store_project_info(project_id, project_data, uploaded_by)
        store_project_id = await self.repo.insert_store_boq_project(store_project_info)
        return store_project_id

    async def _save_location(self, store_project_id: int, location_data: dict) -> int:
        """Save location to database."""
        location_info = self._build_location_info(location_data, store_project_id)
        location_id = await self.repo.insert_location(location_info)
        return location_id

    async def _create_boq_file(self, store_project_id: int, file_info: dict, uploaded_by: str) -> int:
        """Create BOQ file record."""
        boq_file = StoreBoqFile(
            store_project_id=store_project_id,
            file_name=file_info['file_name'],
            file_path=file_info['file_path'],
            file_type=file_info['file_type'],
            created_by=uploaded_by,
        )
        return await self.repo.insert_boq_file(boq_file)

    def _extract_all_items(self, sheets: Dict, boq_id: int, location_id: int) -> List:
        """Extract items from all sheets."""
        all_items = []
        for idx, (sheet_name, sheet_df) in enumerate(sheets.items(), 1):
            items = self._process_sheet(sheet_df, sheet_name, boq_id, location_id)
            all_items.extend(items)
        return all_items

<<<<<<< HEAD
    def _process_sheet(self, sheet_df: pd.DataFrame, sheet_name: str, boq_id: int, location_id: int) -> List:
        """Process single sheet and extract line items."""
        try:
            logger.info(f"  Analyzing sheet structure...")
            sheet_text = self._sheet_to_text(sheet_df, 30)
            structure_json = self.structure_tool.invoke({
                "sheet_data": sheet_text,
                "sheet_name": sheet_name
            })
            structure = json.loads(structure_json)
            
            column_types_found = []
            for col in structure.get('column_structure', []):
                col_type = col['type']
                column_types_found.append(col_type)
            
            required = ['description', 'quantity']
            missing = [r for r in required if r not in column_types_found]
            if missing:
                logger.warning(f"  Missing required columns: {missing}")
                return []
            
            logger.info(f"  Extracting line items...")
            items = self.item_extractor.execute(sheet_df, structure, boq_id, location_id)
            return items
        
        except Exception as e:
            logger.error(f"  Failed to process sheet '{sheet_name}': {e}", exc_info=True)
            return []
        
    async def _generate_embeddings(self, boq_id: int, items: list) -> Dict:
        """Generate embeddings for all items using parallel batch processing."""
        if not items:
            return {"generated": False, "count": 0, "time": 0}
        logger.info("Starting embedding generation...")
        embedding_start = time.time() 
        
        try:
            await self.repo.ensure_vector_extension()
            
            items_to_embed = await self.repo.get_items_without_embeddings(boq_id)
            
            if not items_to_embed:
                logger.info("All items already have embeddings")
                return {"generated": False, "count": 0, "time": 0}
            
<<<<<<< HEAD
            logger.info(f"Generating embeddings for {len(items_to_embed)} items in parallel batches...")
            
            batch_size = 20
            total_generated = 0
            failed_items = []
=======
            logger.info(f"Generating embeddings for {len(items_to_embed)} items")
            
            # Generate embeddings in batches
            batch_size = 100
            total_generated = 0
>>>>>>> 3ecfa74882656e185b462f94606b26887e4e7cc3
            
            for i in range(0, len(items_to_embed), batch_size):
                batch = items_to_embed[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(items_to_embed) + batch_size - 1) // batch_size
                
<<<<<<< HEAD
                logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
                
                result = self._generate_batch_embeddings_with_retry(batch, max_retries=3)
                
                if result['success']:
                    if result['item_ids'] and result['embeddings']:
                        updated = await self.repo.update_embeddings_batch(
                            result['item_ids'], 
                            result['embeddings']
                        )
                        total_generated += updated
                    
                    failed_items.extend(result['failed_item_ids'])
                else:
                    logger.error(f"Entire batch failed: {result.get('error', 'Unknown error')}")
                    failed_items.extend([item.item_id for item in batch])

                if batch_num < total_batches:
                    time.sleep(0.5)
            
            embedding_time = time.time() - embedding_start
            
            logger.info(f"Embedding generation completed")
            if failed_items:
                logger.warning(f"   Failed item IDs: {failed_items[:20]}")
                if len(failed_items) > 20:
                    logger.warning(f"  ... and {len(failed_items) - 20} more")
            
            return {
                "generated": True,
                "count": total_generated,
<<<<<<< HEAD
                "time": embedding_time,
                "failed": len(failed_items)
            }
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}", exc_info=True)
            return {
                "generated": False,
                "count": 0,
                "time": time.time() - embedding_start,
                "error": str(e)
            }

    def _generate_batch_embeddings_with_retry(self, batch: List, max_retries: int = 3) -> Dict:
        """Generate embeddings for a batch of items in parallel with retry logic."""
        batch_descriptions = [item.item_description for item in batch]
        batch_item_ids = [item.item_id for item in batch]
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"    Attempt {attempt + 1}/{max_retries} for batch of {len(batch)} items")
                
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=batch_descriptions,
                    task_type="retrieval_document"
                )
                
                embeddings = result['embedding']
                
                if len(embeddings) != len(batch):
                    logger.warning(
                        f"    Expected {len(batch)} embeddings but got {len(embeddings)}"
                    )
                
                return {
                    'success': True,
                    'item_ids': batch_item_ids[:len(embeddings)],
                    'embeddings': embeddings,
                    'failed_item_ids': batch_item_ids[len(embeddings):] if len(embeddings) < len(batch) else []
                }
                
            except Exception as e:
                error_msg = str(e)
                
                is_retryable = any(keyword in error_msg.lower() for keyword in [
                    '500', 'internal error', 'timeout', 'rate limit', 
                    'quota', 'unavailable', 'deadline', '429', 'too many requests'
                ])
                
                if is_retryable and attempt < max_retries - 1:
                    base_wait = (attempt + 1) * 2
                    jitter = random.uniform(0, 1)
                    wait_time = base_wait + jitter
                    
                    logger.warning(
                        f"    Retry {attempt + 1}/{max_retries} for batch "
                        f"after {wait_time:.1f}s (error: {error_msg[:100]})"
                    )
                    time.sleep(wait_time)
                else:
                    if attempt < max_retries - 1:
                        logger.error(f"Non-retryable error for batch: {error_msg[:200]}")
                    else:
                        logger.error(
                            f"Batch failed after {max_retries} attempts: {error_msg[:200]}"
                        )
                    
                    return {
                        'success': False,
                        'error': error_msg,
                        'item_ids': [],
                        'embeddings': [],
                        'failed_item_ids': batch_item_ids
                    }

        return {
            'success': False,
            'error': 'Unknown error',
            'item_ids': [],
            'embeddings': [],
            'failed_item_ids': batch_item_ids
        }

    async def delete_boq(self, boq_id: int) -> Dict[str, Any]:
        """Delete Store BOQ and all related data."""
        try:
            boq_info = await self.repo.get_boq_info(boq_id)
            if not boq_info:
                return {
                    'success': False, 
                    'error': f'Store BOQ with ID {boq_id} not found'
                }
            
            result = await self.repo.delete_boq_by_id(boq_id)
            
            if result['success']:
                logger.info(f"Successfully deleted BOQ {boq_id}")
                logger.info(f"Deleted counts: {result['deleted_counts']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Delete BOQ error: {e}", exc_info=True)
            return {
                'success': False, 
                'error': f'Failed to delete Store BOQ: {str(e)}'
            }

    async def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get Store BOQ information."""
        try:
            boq_info = await self.repo.get_boq_info(boq_id)
            
            if boq_info:
                logger.info(f"Retrieved info for BOQ {boq_id}")
            else:
                logger.warning(f"BOQ {boq_id} not found")
            
            return boq_info
            
        except Exception as e:
            logger.error(f"Get BOQ info error: {e}", exc_info=True)
            return None

    async def get_boq_summary(self, boq_id: int) -> Optional[Dict]:
        """Get Store BOQ summary with totals."""
        try:
            boq_info = await self.repo.get_boq_info(boq_id)
            if not boq_info:
                return None
            
            totals = await self.repo.get_boq_totals(boq_id)
            
            return {
                **boq_info,
                **totals
            }
            
        except Exception as e:
            logger.error(f"Get BOQ summary error: {e}", exc_info=True)
            return None

    @staticmethod
    def _extract_text(df: pd.DataFrame, max_rows: int) -> str:
        """Extract text from dataframe for AI processing."""
=======
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
>>>>>>> 3ecfa74882656e185b462f94606b26887e4e7cc3
        lines = []
        for _, row in df.head(max_rows).iterrows():
            text = " ".join([str(v) for v in row.values if pd.notna(v)])
            if text.strip():
                lines.append(text)
        return "\n".join(lines)[:5000]

    @staticmethod
    def _sheet_to_text(df: pd.DataFrame, max_rows: int) -> str:
        """Convert sheet to text format for structure analysis."""
        lines = []
        
        headers = [str(col) for col in df.columns if str(col) != 'nan']
        if headers:
            lines.append(f"Headers: {' | '.join(headers)}")
        
        for idx in range(min(max_rows, len(df))):
            row_vals = [str(v) for v in df.iloc[idx].values if pd.notna(v)]
            if row_vals:
                lines.append(f"Row {idx}: {' | '.join(row_vals[:15])}")
        
=======
        return "\n".join(lines)[:5000]

    @staticmethod
    def _sheet_to_text(df: pd.DataFrame, max_rows: int) -> str:
        """Convert sheet to text."""
        lines = []
        for idx in range(min(max_rows, len(df))):
            row_vals = [str(v) for v in df.iloc[idx].values if pd.notna(v)]
            if row_vals:
                lines.append(f"Row {idx}: {' | '.join(row_vals[:10])}")
>>>>>>> 3ecfa74882656e185b462f94606b26887e4e7cc3
        return "\n".join(lines)

    @staticmethod
    def _filter_sheets(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Filter out non-BOQ sheets."""
<<<<<<< HEAD
        skip_keywords = ["summary", "assumption", "note", "index", "cover", "terms", "instruction"]
        
        filtered = {}
        for name, df in sheets.items():
            if any(kw in name.lower() for kw in skip_keywords):
                logger.info(f"  Skipping sheet: '{name}' (keyword match)")
                continue
            
            if len(df) < 5:
                logger.info(f"  Skipping sheet: '{name}' (too few rows: {len(df)})")
                continue
            
            if len(df.columns) < 3:
                logger.info(f"  Skipping sheet: '{name}' (too few columns: {len(df.columns)})")
                continue
            
            filtered[name] = df
        
        return filtered
=======
        skip = ["summary", "assumption", "note", "index", "cover", "terms"]
        return {
            name: df for name, df in sheets.items()
            if not any(kw in name.lower() for kw in skip)
            and len(df) >= 5 and len(df.columns) >= 3
        }
>>>>>>> 3ecfa74882656e185b462f94606b26887e4e7cc3

    @staticmethod
    def _build_project_info(data: dict, uploaded_by: str) -> Project:
        """Build ProjectInfo from extracted data."""
        # Convert string dates to date objects for asyncpg
        start_date = None
        end_date = None
        
        if data.get('start_year'):
            try:
                start_date = date(int(data['start_year']), 1, 1)
            except (ValueError, TypeError):
                start_date = None
        
        if data.get('end_year'):
            try:
                end_date = date(int(data['end_year']), 12, 31)
            except (ValueError, TypeError):
                # Try using start_year if end_year fails
                if data.get('start_year'):
                    try:
                        end_date = date(int(data['start_year']), 12, 31)
                    except (ValueError, TypeError):
                        end_date = None
        
        return Project(
            project_name=data.get("project_name", "BOQ Project"),
            project_code=data.get("project_code", f"PROJ-{datetime.now():%Y%m%d}"),
            project_type="boq",
            client_name=data.get("client_name"),
            start_date=start_date,
            end_date=end_date,
            created_by=uploaded_by,
        )

    @staticmethod
    def _build_store_project_info(project_id: int, data: dict, uploaded_by: str) -> StoreBoqProject:
        """Build StoreBOQProjectInfo from extracted data."""
        project_name = data.get("project_name", "BOQ Project")
        return StoreBoqProject(
            project_id=project_id,
            store_project_name=f"{project_name} - Store",
            store_project_code=f"STORE-{datetime.now():%Y%m%d%H%M}",
            created_by=uploaded_by,
        )

    @staticmethod
    def _build_location_info(data: dict, store_project_id: int) -> StoreBoqLocation:
        """Build LocationInfo from extracted data."""
        address_parts = [data.get("location_name", "Unknown")]
        if data.get("city"):
            address_parts.append(data["city"])
        if data.get("state"):
            address_parts.append(data["state"])
            
        return StoreBoqLocation(
            store_project_id=store_project_id,
            location_name=data.get("location_name", "Unknown"),
            address=", ".join(address_parts),
<<<<<<< HEAD
        )
=======
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
>>>>>>> 3ecfa74882656e185b462f94606b26887e4e7cc3
