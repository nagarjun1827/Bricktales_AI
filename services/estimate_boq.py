# services/estimate_boq.py

import time
import re
import json
import requests
import io
import logging
import asyncio
import pandas as pd
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor
from core.settings import settings
from repositories.estimate_boq import TBEProjectInfo, EstimateBOQProjectInfo, TBELocationInfo, TBEBOQFileInfo, TBEBOQItem
from dto.response_dto.estimate_boq import ItemWithPrice
from repositories.estimate_boq import TBEBOQRepository
from repositories.price import PriceRepository
from agents.gemini_tools import AnalyzeSheetStructureTool, ExtractProjectInfoTool, ExtractLocationInfoTool
from agents.item_extractor import ItemExtractorAgent

logger = logging.getLogger(__name__)

class TBEBOQProcessor:
    """Processes To-Be-Estimated BOQ files from URL with automatic price fetching."""
    
    def __init__(self):
        self.repo = TBEBOQRepository()
        self.price_repo = PriceRepository()
        self.item_extractor = ItemExtractorAgent()
        self.project_tool = ExtractProjectInfoTool()
        self.location_tool = ExtractLocationInfoTool()
        self.structure_tool = AnalyzeSheetStructureTool()
        
        # Configure Gemini for embeddings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "models/text-embedding-004"
    
    async def process_file_from_url(
        self, 
        file_url: str, 
        uploaded_by: str = "user",
        top_k: int = 1,
        min_similarity: float = 0.5
    ) -> Dict[str, Any]:
        """Process TBE BOQ file from URL with automatic price fetching."""
        logger.info(f"Estimate file URL: {file_url}")
        logger.info(f"Top K: {top_k}, Min Similarity: {min_similarity}")
        
        start_time = time.time()
        
        try:
            logger.info("Step 1: Downloading and reading Excel file from URL")
            file_start = time.time()
            
            tbe_result = await self._process_tbe_file_from_url(file_url, uploaded_by)
            
            if not tbe_result['success']:
                return tbe_result
            
            file_time = time.time() - file_start
            logger.info(f"File processing completed in {file_time:.2f}s")
            
            boq_id = tbe_result['boq_id']
            project_id = tbe_result['project_id']
            estimate_project_id = tbe_result['estimate_project_id']
            location_id = tbe_result['location_id']
            total_items = tbe_result['total_items']
            
            # STEP 2: Generate Embeddings for Items
            logger.info("Step 2: Generating Embeddings for Items")
            embedding_start = time.time()
            
            embedding_result = await self._generate_embeddings_for_items(boq_id)
            
            if not embedding_result['success']:
                return {
                    'success': False,
                    'error': embedding_result['error'],
                    'error_step': 'embedding_generation'
                }
            
            embedding_time = time.time() - embedding_start
            logger.info(f"Embedding generation completed in {embedding_time:.2f}s")
            
            # STEP 3: Fetch Prices Using Parallel Similarity Search
            logger.info("Step 3: Fetching Prices Using Parallel Similarity Search")
            price_start = time.time()
            
            price_result = await self._fetch_prices_for_items_parallel(
                boq_id, top_k, min_similarity
            )
            
            if not price_result['success']:
                return {
                    'success': False,
                    'error': price_result['error'],
                    'error_step': 'price_fetching'
                }
            
            price_time = time.time() - price_start
            logger.info(f"Price fetching completed in {price_time:.2f}s")
            
            # Calculate totals
            total_estimated_supply = sum(
                item.estimated_supply_total for item in price_result['items']
                if item.estimated_supply_total
            )
            total_estimated_labour = sum(
                item.estimated_labour_total for item in price_result['items']
                if item.estimated_labour_total
            )
            total_estimated_amount = sum(
                item.estimated_total for item in price_result['items']
                if item.estimated_total
            )
            
            total_time = time.time() - start_time
            
            # Log final summary
            self._log_final_summary(
                project_id, estimate_project_id, location_id, boq_id,
                total_items, price_result['items_with_prices'],
                price_result['items_without_prices'],
                total_estimated_supply, total_estimated_labour,
                total_estimated_amount, file_time, embedding_time,
                price_time, total_time
            )
            
            return {
                'success': True,
                'project_id': project_id,
                'estimate_project_id': estimate_project_id,
                'location_id': location_id,
                'boq_id': boq_id,
                'total_items': total_items,
                'items_with_prices': price_result['items_with_prices'],
                'items_without_prices': price_result['items_without_prices'],
                'total_estimated_supply': total_estimated_supply,
                'total_estimated_labour': total_estimated_labour,
                'total_estimated_amount': total_estimated_amount,
                'items': price_result['items'],
                'file_processing_time': file_time,
                'embedding_generation_time': embedding_time,
                'price_fetching_time': price_time,
                'total_processing_time': total_time,
                'message': 'TBE BOQ processed successfully with prices from URL'
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    async def _process_tbe_file_from_url(self, file_url: str, uploaded_by: str) -> Dict[str, Any]:
        """Process TBE BOQ file from URL and extract items."""
        try:
            # Read file from URL
            logger.info("Downloading and reading Excel file from URL")
            sheets = self._read_excel_from_url(file_url)
            logger.info(f"Read {len(sheets)} sheets")
            
            # Extract project information
            first_sheet = list(sheets.values())[0]
            project_data = self._extract_project_info(first_sheet)
            logger.info(f"Project Name: {project_data['project_name']}")
            
            # Extract location information
            location_data = self._extract_location_info(first_sheet)
            logger.info(f"Location: {location_data['location_name']}")
            
            # Save project to database
            project_id = await self._save_project(project_data, uploaded_by)
            logger.info(f"Project ID: {project_id}")
            
            # Save estimate project to database
            estimate_project_id = await self._save_estimate_project(project_id, project_data, uploaded_by)
            logger.info(f"Estimate Project ID: {estimate_project_id}")
            
            # Save location to database
            location_id = await self._save_location(estimate_project_id, location_data)
            logger.info(f"Location ID: {location_id}")
            
            # Extract file information and create BOQ file record
            file_info = self._extract_file_info(file_url)
            boq_id = await self._create_boq_file(estimate_project_id, file_info, uploaded_by)
            logger.info(f"BOQ ID: {boq_id}")
            
            # Filter and process sheets
            boq_sheets = self._filter_sheets(sheets)
            logger.info(f"Found {len(boq_sheets)} BOQ sheets to process")
            for sheet_name in boq_sheets.keys():
                logger.info(f"  - {sheet_name}")
            
            # Extract line items from all sheets
            all_items = self._extract_all_items(boq_sheets, boq_id, location_id)
            logger.info(f"Total items extracted: {len(all_items)}")
            
            if not all_items:
                return {
                    'success': False, 
                    'error': 'No items extracted from BOQ file'
                }
            
            # Insert items with error handling
            try:
                await self.repo.insert_tbe_items_batch(all_items)
                logger.info("Items inserted successfully")
            except Exception as insert_error:
                logger.error(f"Insert failed: {insert_error}", exc_info=True)
                return {
                    'success': False,
                    'error': f'Failed to insert items: {str(insert_error)}'
                }
            
            # Summary
            try:
                summary = await self.repo.get_tbe_boq_summary(boq_id)
                logger.info(f"Summary retrieved: {summary['item_count']} items")
            except Exception as summary_error:
                logger.warning(f"Summary retrieval failed: {summary_error}")
                summary = {'item_count': len(all_items)}
            
            return {
                'success': True,
                'project_id': project_id,
                'estimate_project_id': estimate_project_id,
                'location_id': location_id,
                'boq_id': boq_id,
                'total_items': summary['item_count']
            }
            
        except Exception as e:
            logger.error(f"File processing error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _read_excel_from_url(self, file_url: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file at URL."""
        try:
            logger.info("Fetching file from URL")
            response = requests.get(file_url, timeout=120)
            response.raise_for_status()
            
            logger.info(f"File downloaded successfully ({len(response.content):,} bytes)")
            
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
                "project_name": f"TBE Project {datetime.now():%Y%m%d}",
                "project_code": f"TBE-{datetime.now():%Y%m%d%H%M}",
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
        try:
            from urllib.parse import urlparse
            parsed = urlparse(file_url)
            filename = parsed.path.split('/')[-1] or "estimate_boq.xlsx"
            
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
                "file_name": "estimate_boq.xlsx",
                "file_path": file_url,
                "file_type": "xlsx"
            }
    
    async def _save_project(self, project_data: dict, uploaded_by: str) -> int:
        """Save project to database."""
        project_info = self._build_project_info(project_data, uploaded_by)
        project_id = await self.repo.insert_project(project_info)
        return project_id
    
    async def _save_estimate_project(self, project_id: int, project_data: dict, uploaded_by: str) -> int:
        """Save estimate project to database."""
        estimate_project_info = self._build_estimate_project_info(project_id, project_data, uploaded_by)
        estimate_project_id = await self.repo.insert_estimate_boq_project(estimate_project_info)
        return estimate_project_id
    
    async def _save_location(self, estimate_project_id: int, location_data: dict) -> int:
        """Save location to database."""
        location_info = self._build_location_info(location_data, estimate_project_id)
        location_id = await self.repo.insert_location(location_info)
        return location_id
    
    async def _create_boq_file(self, estimate_project_id: int, file_info: dict, uploaded_by: str) -> int:
        """Create BOQ file record."""
        boq_file = TBEBOQFileInfo(
            estimate_project_id=estimate_project_id,
            file_name=file_info['file_name'],
            file_path=file_info['file_path'],
            created_by=uploaded_by,
        )
        return await self.repo.insert_tbe_boq_file(boq_file)
    
    def _extract_all_items(self, sheets: Dict, boq_id: int, location_id: int) -> List[TBEBOQItem]:
        """Extract items from all sheets."""
        all_items = []
        for idx, (sheet_name, sheet_df) in enumerate(sheets.items(), 1):
            items = self._process_sheet(sheet_df, sheet_name, boq_id, location_id)
            all_items.extend(items)
        return all_items
    
    def _process_sheet(self, sheet_df: pd.DataFrame, sheet_name: str, boq_id: int, location_id: int) -> List[TBEBOQItem]:
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
    
    async def _generate_embeddings_for_items(self, boq_id: int) -> Dict[str, Any]:
        """Generate embeddings for all items in TBE BOQ using parallel batch processing."""
        try:
            items = await self.repo.get_tbe_items_by_boq(boq_id, limit=10000, offset=0)
            
            if not items:
                return {'success': False, 'error': 'No items found to generate embeddings'}
            
            logger.info(f"Generating embeddings for {len(items)} items in parallel batches...")
            
            batch_size = 20
            embeddings_map = {}
            failed_items = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(items) + batch_size - 1) // batch_size
                
                logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} items)")
                
                batch_descriptions = [item['item_description'] for item in batch]
                batch_item_ids = [item['item_id'] for item in batch]
                
                try:
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=batch_descriptions,
                        task_type="retrieval_query"
                    )
                    
                    embeddings = result['embedding']
                    for item_id, embedding in zip(batch_item_ids, embeddings):
                        embeddings_map[item_id] = embedding
                    
                    logger.info(f"Generated {len(embeddings)} embeddings")
                    
                except Exception as e:
                    logger.warning(f"Batch {batch_num} failed: {e}")
                    failed_items.extend(batch_item_ids)
                    continue
                
                if batch_num < total_batches:
                    time.sleep(0.5)
            
            logger.info(f"Total embeddings generated: {len(embeddings_map)}")
            if failed_items:
                logger.warning(f"Failed items: {len(failed_items)}")
            
            self._embeddings_cache = embeddings_map
            
            return {
                'success': True,
                'count': len(embeddings_map),
                'failed': len(failed_items)
            }
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    async def _fetch_prices_for_items_parallel(
        self, 
        boq_id: int, 
        top_k: int, 
        min_similarity: float,
        batch_size: int = 20,
        max_workers: int = 10
    ) -> Dict[str, Any]:
        """
        Fetch prices for all items using parallel similarity search.
        
        Args:
            boq_id: BOQ ID to fetch items for
            top_k: Number of similar items to fetch per item
            min_similarity: Minimum similarity threshold
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of parallel workers
        
        Returns:
            Dict with success status, counts, and item results
        """
        try:
            items = await self.repo.get_tbe_items_by_boq(boq_id, limit=10000, offset=0)
            
            if not items:
                return {'success': False, 'error': 'No items found'}
            
            logger.info(f"Fetching prices for {len(items)} items in parallel batches")
            logger.info(f"Batch size: {batch_size}, Max workers: {max_workers}")
            
            items_with_prices = 0
            items_without_prices = 0
            result_items = []
            
            # Process items in batches
            for batch_start in range(0, len(items), batch_size):
                batch_end = min(batch_start + batch_size, len(items))
                batch = items[batch_start:batch_end]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(items) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
                
                # Create tasks for parallel processing
                tasks = []
                for item in batch:
                    task = self._fetch_price_for_single_item(
                        item, 
                        top_k, 
                        min_similarity
                    )
                    tasks.append(task)
                
                # Execute all tasks in parallel
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for idx, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing item: {result}")
                        items_without_prices += 1
                        # Create error result item
                        item = batch[idx]
                        result_items.append(ItemWithPrice(
                            item_id=item['item_id'],
                            item_code=item['item_code'],
                            description=item['item_description'],
                            unit=item['unit_of_measurement'],
                            quantity=item['quantity'],
                            estimated_supply_rate=None,
                            estimated_labour_rate=None,
                            estimated_supply_total=None,
                            estimated_labour_total=None,
                            estimated_total=None,
                            pricing_source=f"Error: {str(result)}",
                            similarity_score=None
                        ))
                    else:
                        if result['has_price']:
                            items_with_prices += 1
                        else:
                            items_without_prices += 1
                        result_items.append(result['item'])
                
                logger.info(f"Batch {batch_num} completed: {items_with_prices} with prices, {items_without_prices} without")
            
            # Clean up embeddings cache
            if hasattr(self, '_embeddings_cache'):
                del self._embeddings_cache
            
            logger.info(f"Price fetching completed: {items_with_prices} with prices, {items_without_prices} without")
            
            return {
                'success': True,
                'items_with_prices': items_with_prices,
                'items_without_prices': items_without_prices,
                'items': result_items
            }
            
        except Exception as e:
            logger.error(f"Price fetching error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    async def _fetch_price_for_single_item(
        self, 
        item: Dict, 
        top_k: int, 
        min_similarity: float
    ) -> Dict[str, Any]:
        """
        Fetch price for a single item using similarity search.
        
        Args:
            item: Item dictionary with metadata
            top_k: Number of similar items to fetch
            min_similarity: Minimum similarity threshold
        
        Returns:
            Dict with has_price flag and ItemWithPrice object
        """
        item_id = item['item_id']
        description = item['item_description']
        unit = item['unit_of_measurement']
        quantity = item['quantity']
        item_code = item['item_code']
        
        # Check if embedding exists
        if not hasattr(self, '_embeddings_cache') or item_id not in self._embeddings_cache:
            logger.warning(f"No embedding found for item {item_id}")
            return {
                'has_price': False,
                'item': ItemWithPrice(
                    item_id=item_id,
                    item_code=item_code,
                    description=description,
                    unit=unit,
                    quantity=quantity,
                    estimated_supply_rate=None,
                    estimated_labour_rate=None,
                    estimated_supply_total=None,
                    estimated_labour_total=None,
                    estimated_total=None,
                    pricing_source="No embedding generated",
                    similarity_score=None
                )
            }
        
        query_embedding = self._embeddings_cache[item_id]
        
        # Find similar items
        similar_items = await self.price_repo.find_similar_items(
            query_embedding=query_embedding,
            unit=unit,
            limit=top_k,
            min_similarity=min_similarity
        )
        
        if not similar_items:
            return {
                'has_price': False,
                'item': ItemWithPrice(
                    item_id=item_id,
                    item_code=item_code,
                    description=description,
                    unit=unit,
                    quantity=quantity,
                    estimated_supply_rate=None,
                    estimated_labour_rate=None,
                    estimated_supply_total=None,
                    estimated_labour_total=None,
                    estimated_total=None,
                    pricing_source=f"No similar items found (min similarity: {min_similarity})",
                    similarity_score=None
                )
            }
        
        # Get best match
        best_match = similar_items[0]
        similarity_score = best_match['similarity']
        
        # Extract rates
        estimated_supply_rate = None
        estimated_labour_rate = None
        
        if best_match.get('supply_rate') and float(best_match['supply_rate']) > 0:
            estimated_supply_rate = float(best_match['supply_rate'])
        
        if best_match.get('labour_rate') and float(best_match['labour_rate']) > 0:
            estimated_labour_rate = float(best_match['labour_rate'])
        
        # Calculate totals
        estimated_supply_total = None
        estimated_labour_total = None
        estimated_total = 0.0
        
        if estimated_supply_rate:
            estimated_supply_total = estimated_supply_rate * quantity
            estimated_total += estimated_supply_total
        
        if estimated_labour_rate:
            estimated_labour_total = estimated_labour_rate * quantity
            estimated_total += estimated_labour_total
        
        if estimated_supply_total is None and estimated_labour_total is None:
            estimated_total = None
        
        # Build pricing source
        pricing_source = self._build_pricing_source(best_match, similarity_score)
        
        return {
            'has_price': True,
            'item': ItemWithPrice(
                item_id=item_id,
                item_code=item_code,
                description=description,
                unit=unit,
                quantity=quantity,
                estimated_supply_rate=estimated_supply_rate,
                estimated_labour_rate=estimated_labour_rate,
                estimated_supply_total=estimated_supply_total,
                estimated_labour_total=estimated_labour_total,
                estimated_total=estimated_total,
                pricing_source=pricing_source,
                similarity_score=round(similarity_score, 3)
            )
        }
    
    @staticmethod
    def _parse_date_from_year(year: Optional[int], month: int = 1, day: int = 1) -> Optional[date]:
        """Parse date from year, handling None and invalid values."""
        if not year:
            return None
        try:
            return date(int(year), month, day)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _build_pricing_source(match: dict, similarity: float) -> str:
        """Build a descriptive pricing source explanation."""
        project = match.get('project_name', 'Unknown Project')
        item_desc = match.get('description', 'Unknown Item')
        item_code = match.get('item_code', 'N/A')
        
        if len(item_desc) > 50:
            item_desc = item_desc[:47] + "..."
        
        source = f"Matched with '{item_desc}' (Code: {item_code}) from project '{project}' (Similarity: {similarity:.1%})"
        
        return source
    
    @staticmethod
    def _extract_text(df: pd.DataFrame, max_rows: int) -> str:
        """Extract text from dataframe for AI processing."""
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
        
        return "\n".join(lines)
    
    @staticmethod
    def _filter_sheets(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Filter out non-BOQ sheets."""
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
    
    @staticmethod
    def _build_project_info(data: dict, uploaded_by: str) -> TBEProjectInfo:
        """Build project info."""
        start_year = data.get('start_year')
        end_year = data.get('end_year')
        
        # Convert years to date objects for asyncpg compatibility
        start_date = TBEBOQProcessor._parse_date_from_year(start_year, month=1, day=1)
        end_date = TBEBOQProcessor._parse_date_from_year(
            end_year or start_year, 
            month=12, 
            day=31
        )
        
        return TBEProjectInfo(
            project_name=data.get("project_name", f"TBE Project {datetime.now():%Y%m%d}"),
            project_code=data.get("project_code", f"TBE-{datetime.now():%Y%m%d%H%M}"),
            project_type="boq",
            client_name=data.get("client_name"),
            start_date=start_date,
            end_date=end_date,
            created_by=uploaded_by,
        )
    
    @staticmethod
    def _build_estimate_project_info(
        project_id: int, 
        data: dict, 
        uploaded_by: str
    ) -> EstimateBOQProjectInfo:
        """Build estimate project info."""
        project_name = data.get("project_name", f"TBE Project {datetime.now():%Y%m%d}")
        return EstimateBOQProjectInfo(
            project_id=project_id,
            estimate_project_name=f"{project_name} - Estimate",
            estimate_project_code=f"EST-{datetime.now():%Y%m%d%H%M}",
            estimation_status="pending",
            created_by=uploaded_by,
        )
    
    @staticmethod
    def _build_location_info(data: dict, estimate_project_id: int) -> TBELocationInfo:
        """Build location info."""
        address_parts = [data.get("location_name", "Unknown")]
        if data.get("city"):
            address_parts.append(data["city"])
        if data.get("state"):
            address_parts.append(data["state"])
            
        return TBELocationInfo(
            estimate_project_id=estimate_project_id,
            location_name=data.get("location_name", "Unknown"),
            address=", ".join(address_parts)
        )
    
    @staticmethod
    def _log_final_summary(
        project_id, estimate_project_id, location_id, boq_id,
        total_items, items_with_prices, items_without_prices,
        total_estimated_supply, total_estimated_labour,
        total_estimated_amount, file_time, embedding_time,
        price_time, total_time
    ):
        """Log final processing summary."""
        logger.info("Complete Processing Summary:")
        logger.info(f"Project ID: {project_id}")
        logger.info(f"Estimate Project ID: {estimate_project_id}")
        logger.info(f"Location ID: {location_id}")
        logger.info(f"BOQ ID: {boq_id}")
        logger.info(f"Total Items: {total_items}")
        logger.info(f"Items with Prices: {items_with_prices}")
        logger.info(f"Items without Prices: {items_without_prices}")
        logger.info(f"Supply Amount: ₹{total_estimated_supply:,.2f}")
        logger.info(f"Labour Amount: ₹{total_estimated_labour:,.2f}")
        logger.info(f"Total Amount: ₹{total_estimated_amount:,.2f}")
        logger.info(f"File Processing: {file_time:.2f}s")
        logger.info(f"Embedding Generation: {embedding_time:.2f}s")
        logger.info(f"Price Fetching: {price_time:.2f}s")
        logger.info(f"Total Time: {total_time:.2f}s")