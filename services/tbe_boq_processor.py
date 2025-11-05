"""
Service for processing To-Be-Estimated BOQ files from URL with automatic price fetching.
"""
import time
import re
import json
import requests
import io
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import google.generativeai as genai

from core.settings import settings
from models.tbe_domain import TBEProjectInfo, EstimateBOQProjectInfo, TBELocationInfo, TBEBOQFileInfo, TBEBOQItem
from models.tbe_dto import ItemWithPrice
from repositories.tbe_boq_repository import TBEBOQRepository
from repositories.price_repository import PriceRepository
from services.pattern_matcher import PatternMatcher


class GeminiExtractor:
    """Gemini AI extractor for TBE BOQ."""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def extract_project_info(self, text: str) -> Dict:
        """Extract project information."""
        prompt = f"""Extract project info from: {text}
Return JSON: {{"project_name": str, "project_code": str, "year": str, "location": str, "client_name": str|null}}"""

        try:
            response = self.model.generate_content(prompt)
            return json.loads(self._clean_json(response.text))
        except:
            return {}
    
    def identify_columns(self, columns: List[str]) -> Dict:
        """Identify column mappings."""
        cols = ', '.join([f'"{c}"' for c in columns])
        prompt = f"""Map columns to: item_code, description, quantity, unit
Columns: {cols}
Return JSON mapping."""

        try:
            response = self.model.generate_content(prompt)
            return json.loads(self._clean_json(response.text))
        except:
            return {}
    
    @staticmethod
    def _clean_json(text: str) -> str:
        """Clean JSON response."""
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return text.strip()


class TBEBOQProcessor:
    """Processes To-Be-Estimated BOQ files from URL with automatic price fetching."""
    
    def __init__(self):
        self.repo = TBEBOQRepository()
        self.price_repo = PriceRepository()
        self.extractor = GeminiExtractor()
        self.pattern_matcher = PatternMatcher()
        
        # Configure Gemini for embeddings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "models/text-embedding-004"
    
    def process_file_from_url(
        self, 
        file_url: str, 
        uploaded_by: str = "system",
        top_k: int = 1,
        min_similarity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process TBE BOQ file from URL with automatic price fetching.
        
        Workflow:
        1. Download and process file from URL
        2. Generate embeddings for items
        3. Fetch prices using similarity search (top 1 match only)
        4. Return complete results with pricing source
        """
        print(f"\n{'='*80}")
        print(f"🚀 TBE BOQ PROCESSOR FROM URL WITH AUTOMATIC PRICE FETCHING")
        print(f"{'='*80}")
        print(f"URL: {file_url}")
        print(f"Top K: {top_k}, Min Similarity: {min_similarity}\n")
        
        start_time = time.time()
        
        try:
            # STEP 1: Process TBE BOQ File from URL
            print("=" * 80)
            print("STEP 1: PROCESSING TBE BOQ FILE FROM URL")
            print("=" * 80)
            file_start = time.time()
            
            tbe_result = self._process_tbe_file_from_url(file_url, uploaded_by)
            
            if not tbe_result['success']:
                return tbe_result
            
            file_time = time.time() - file_start
            print(f"\n✓ File processing completed in {file_time:.2f}s\n")
            
            boq_id = tbe_result['boq_id']
            project_id = tbe_result['project_id']
            estimate_project_id = tbe_result['estimate_project_id']
            location_id = tbe_result['location_id']
            total_items = tbe_result['total_items']
            
            # STEP 2: Generate Embeddings for Items
            print("=" * 80)
            print("STEP 2: GENERATING EMBEDDINGS FOR ITEMS")
            print("=" * 80)
            embedding_start = time.time()
            
            embedding_result = self._generate_embeddings_for_items(boq_id)
            
            if not embedding_result['success']:
                return {
                    'success': False,
                    'error': embedding_result['error'],
                    'error_step': 'embedding_generation'
                }
            
            embedding_time = time.time() - embedding_start
            print(f"\n✓ Embedding generation completed in {embedding_time:.2f}s\n")
            
            # STEP 3: Fetch Prices Using Similarity Search
            print("=" * 80)
            print("STEP 3: FETCHING PRICES USING SIMILARITY SEARCH (TOP 1)")
            print("=" * 80)
            price_start = time.time()
            
            price_result = self._fetch_prices_for_items(
                boq_id, top_k, min_similarity
            )
            
            if not price_result['success']:
                return {
                    'success': False,
                    'error': price_result['error'],
                    'error_step': 'price_fetching'
                }
            
            price_time = time.time() - price_start
            print(f"\n✓ Price fetching completed in {price_time:.2f}s\n")
            
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
            
            # Print final summary
            self._print_final_summary(
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
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _process_tbe_file_from_url(self, file_url: str, uploaded_by: str) -> Dict[str, Any]:
        """Process TBE BOQ file from URL and extract items."""
        try:
            # Read file from URL
            print("📥 Downloading and reading Excel file from URL...")
            sheets = self._read_excel_from_url(file_url)
            print(f"   ✓ Read {len(sheets)} sheets\n")
            
            # Extract metadata
            first_sheet = list(sheets.values())[0]
            project_id, estimate_project_id, location_id = self._process_metadata(
                first_sheet, uploaded_by
            )
            
            # Create BOQ file
            print("💾 Creating TBE BOQ file record...")
            boq_id = self._create_boq_file(estimate_project_id, file_url, uploaded_by)
            print(f"   ✓ BOQ ID: {boq_id}\n")
            
            # Extract items
            print("📊 Extracting items...")
            all_items = []
            for name, df in sheets.items():
                if self._should_skip_sheet(name, df):
                    continue
                print(f"   📄 {name}")
                items = self._extract_items(df, boq_id, location_id)
                all_items.extend(items)
            
            # Insert items with error handling
            print(f"\n💾 Inserting {len(all_items)} items...")
            
            if not all_items:
                return {
                    'success': False, 
                    'error': 'No items extracted from BOQ file'
                }
            
            try:
                self.repo.insert_tbe_items_batch(all_items)
                print(f"   ✓ Items inserted successfully\n")
            except Exception as insert_error:
                print(f"   ✗ Insert failed: {insert_error}")
                import traceback
                traceback.print_exc()
                return {
                    'success': False,
                    'error': f'Failed to insert items: {str(insert_error)}'
                }
            
            # Summary
            try:
                summary = self.repo.get_tbe_boq_summary(boq_id)
                print(f"   ✓ Summary retrieved: {summary['item_count']} items\n")
            except Exception as summary_error:
                print(f"   ⚠️  Summary retrieval failed: {summary_error}")
                # Use fallback count
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
            print(f"\n✗ File processing error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _read_excel_from_url(self, file_url: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file at URL."""
        try:
            print(f"   Fetching file from URL...")
            response = requests.get(file_url, timeout=120)
            response.raise_for_status()
            
            print(f"   File downloaded ({len(response.content)} bytes)")
            print(f"   Reading Excel sheets...")
            
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
    
    def _generate_embeddings_for_items(self, boq_id: int) -> Dict[str, Any]:
        """Generate embeddings for all items in TBE BOQ."""
        try:
            # Get all items
            items = self.repo.get_tbe_items_by_boq(boq_id, limit=10000, offset=0)
            
            if not items:
                return {'success': False, 'error': 'No items found to generate embeddings'}
            
            print(f"📝 Generating embeddings for {len(items)} items...")
            
            batch_size = 100
            embeddings_map = {}  # item_id -> embedding
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(items) + batch_size - 1) // batch_size
                
                print(f"   Batch {batch_num}/{total_batches} ({len(batch)} items)...")
                
                for item in batch:
                    try:
                        result = genai.embed_content(
                            model=self.embedding_model,
                            content=item['item_description'],
                            task_type="retrieval_query"
                        )
                        embeddings_map[item['item_id']] = result['embedding']
                    except Exception as e:
                        print(f"      ⚠ Failed for item {item['item_id']}: {e}")
                        continue
                
                print(f"   ✓ Generated {len(batch)} embeddings")
            
            print(f"\n✓ Total embeddings generated: {len(embeddings_map)}")
            
            # Store embeddings temporarily in class instance for price fetching
            self._embeddings_cache = embeddings_map
            
            return {
                'success': True,
                'count': len(embeddings_map)
            }
            
        except Exception as e:
            print(f"\n✗ Embedding generation error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _fetch_prices_for_items(
        self, 
        boq_id: int, 
        top_k: int, 
        min_similarity: float
    ) -> Dict[str, Any]:
        """Fetch prices for all items using similarity search (top 1 match only)."""
        try:
            # Get all items
            items = self.repo.get_tbe_items_by_boq(boq_id, limit=10000, offset=0)
            
            if not items:
                return {'success': False, 'error': 'No items found'}
            
            print(f"🔍 Fetching prices for {len(items)} items (top 1 match per item)...\n")
            
            items_with_prices = 0
            items_without_prices = 0
            result_items = []
            
            for idx, item in enumerate(items, 1):
                item_id = item['item_id']
                description = item['item_description']
                unit = item['unit_of_measurement']
                quantity = item['quantity']
                item_code = item['item_code']
                
                print(f"[{idx}/{len(items)}] {item_code or 'N/A'}: {description[:60]}...")
                
                # Get embedding from cache
                if not hasattr(self, '_embeddings_cache') or item_id not in self._embeddings_cache:
                    print(f"  ⚠️  No embedding found, skipping")
                    items_without_prices += 1
                    result_items.append(ItemWithPrice(
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
                    ))
                    continue
                
                query_embedding = self._embeddings_cache[item_id]
                
                # Find similar items (top 1 only)
                similar_items = self.price_repo.find_similar_items(
                    query_embedding=query_embedding,
                    unit=unit,
                    limit=top_k,  # Always 1
                    min_similarity=min_similarity
                )
                
                if similar_items:
                    # Take only the first (best) match
                    best_match = similar_items[0]
                    
                    similarity_score = best_match['similarity']
                    print(f"  ✓ Found match (similarity: {similarity_score:.3f})")
                    items_with_prices += 1
                    
                    # Use rates from the best match directly
                    estimated_supply_rate = float(best_match['supply_rate']) if best_match['supply_rate'] > 0 else None
                    estimated_labour_rate = float(best_match['labour_rate']) if best_match['labour_rate'] > 0 else None
                    
                    # Estimated totals
                    estimated_supply_total = estimated_supply_rate * quantity if estimated_supply_rate else None
                    estimated_labour_total = estimated_labour_rate * quantity if estimated_labour_rate else None
                    estimated_total = None
                    if estimated_supply_total is not None and estimated_labour_total is not None:
                        estimated_total = estimated_supply_total + estimated_labour_total
                    elif estimated_supply_total is not None:
                        estimated_total = estimated_supply_total
                    
                    # Build pricing source explanation
                    pricing_source = self._build_pricing_source(best_match, similarity_score)
                    
                    if estimated_supply_rate:
                        print(f"  💰 Supply Rate: ₹{estimated_supply_rate:,.2f}/{unit}")
                    if estimated_labour_rate:
                        print(f"  💰 Labour Rate: ₹{estimated_labour_rate:,.2f}/{unit}")
                    print(f"  📋 Source: {pricing_source}")
                    
                    result_items.append(ItemWithPrice(
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
                    ))
                else:
                    print(f"  ⚠️  No similar items found")
                    items_without_prices += 1
                    result_items.append(ItemWithPrice(
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
                    ))
            
            # Clean up cache
            if hasattr(self, '_embeddings_cache'):
                del self._embeddings_cache
            
            return {
                'success': True,
                'items_with_prices': items_with_prices,
                'items_without_prices': items_without_prices,
                'items': result_items
            }
            
        except Exception as e:
            print(f"\n✗ Price fetching error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _build_pricing_source(match: dict, similarity: float) -> str:
        """Build a descriptive pricing source explanation."""
        project = match.get('project_name', 'Unknown Project')
        file = match.get('file_name', 'Unknown File')
        item_desc = match.get('description', 'Unknown Item')
        item_code = match.get('item_code', 'N/A')
        
        # Truncate description if too long
        if len(item_desc) > 50:
            item_desc = item_desc[:47] + "..."
        
        source = f"Matched with '{item_desc}' (Code: {item_code}) from project '{project}' (Similarity: {similarity:.1%})"
        
        return source
    
    def _process_metadata(self, df: pd.DataFrame, uploaded_by: str) -> tuple:
        """Extract and save project/estimate project/location."""
        text = df.head(20).to_string()
        
        # Extract project
        project_data = self.extractor.extract_project_info(text)
        project_info = self._build_project_info(project_data, uploaded_by)
        project_id = self.repo.insert_project(project_info)
        print(f"   ✓ Project ID: {project_id}")
        
        # Create estimate project
        estimate_project_info = self._build_estimate_project_info(
            project_id, project_data, uploaded_by
        )
        estimate_project_id = self.repo.insert_estimate_boq_project(estimate_project_info)
        print(f"   ✓ Estimate Project ID: {estimate_project_id}")
        
        # Extract location
        location_info = self._build_location_info(project_data, estimate_project_id)
        location_id = self.repo.insert_location(location_info)
        print(f"   ✓ Location ID: {location_id}")
        
        return project_id, estimate_project_id, location_id
    
    def _create_boq_file(self, estimate_project_id: int, file_url: str, uploaded_by: str) -> int:
        """Create BOQ file record."""
        # Extract filename from URL or use generic name
        try:
            from urllib.parse import urlparse
            parsed = urlparse(file_url)
            filename = parsed.path.split('/')[-1] or "estimate_boq_from_url.xlsx"
        except:
            filename = "estimate_boq_from_url.xlsx"
            
        file_info = TBEBOQFileInfo(
            estimate_project_id=estimate_project_id,
            file_name=filename,
            file_path=file_url,  # Store URL instead of local path
            created_by=uploaded_by
        )
        return self.repo.insert_tbe_boq_file(file_info)
    
    def _extract_items(self, df: pd.DataFrame, boq_id: int, location_id: int) -> List[TBEBOQItem]:
        """Extract items from sheet."""
        # Identify columns
        col_mapping = self.pattern_matcher.identify_columns(df)
        
        if not col_mapping['description']:
            gemini_mapping = self.extractor.identify_columns(df.columns.tolist())
            for key, value in gemini_mapping.items():
                if value and not col_mapping.get(key):
                    col_mapping[key] = value
        
        # Extract items
        items = []
        for _, row in df.iterrows():
            if col_mapping['description'] and pd.isna(row[col_mapping['description']]):
                continue
            
            # Get values
            item_code = str(row[col_mapping['item_code']]) if col_mapping['item_code'] and pd.notna(row[col_mapping['item_code']]) else None
            desc = str(row[col_mapping['description']]) if col_mapping['description'] and pd.notna(row[col_mapping['description']]) else ""
            quantity = self.pattern_matcher.extract_numeric(row[col_mapping['quantity']]) if col_mapping['quantity'] else 0
            unit = self.pattern_matcher.normalize_unit(row[col_mapping['unit']]) if col_mapping['unit'] else "Each"
            
            if quantity == 0 or not desc:
                continue
            
            items.append(TBEBOQItem(
                boq_id=boq_id,
                item_code=item_code,
                item_description=re.sub(r'\s+', ' ', desc).strip(),
                unit_of_measurement=unit,
                quantity=quantity,
                location_id=location_id
            ))
        
        print(f"      ✓ Extracted {len(items)} items")
        return items
    
    @staticmethod
    def _should_skip_sheet(name: str, df: pd.DataFrame) -> bool:
        """Check if sheet should be skipped."""
        skip = ['summary', 'assumption', 'note', 'index', 'cover']
        return any(kw in name.lower() for kw in skip) or len(df) < 5
    
    @staticmethod
    def _build_project_info(data: dict, uploaded_by: str) -> TBEProjectInfo:
        """Build project info."""
        year = data.get('year', datetime.now().year)
        return TBEProjectInfo(
            project_name=data.get('project_name', f"TBE Project {datetime.now():%Y%m%d}"),
            project_code=data.get('project_code', f"TBE-{year}-{datetime.now():%m%d%H%M}"),
            project_type="boq",
            client_name=data.get('client_name'),
            start_date=f"{year}-01-01" if year else None,
            end_date=f"{year}-12-31" if year else None,
            created_by=uploaded_by,
        )
    
    @staticmethod
    def _build_estimate_project_info(
        project_id: int, 
        data: dict, 
        uploaded_by: str
    ) -> EstimateBOQProjectInfo:
        """Build estimate project info."""
        project_name = data.get('project_name', f"TBE Project {datetime.now():%Y%m%d}")
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
        location = data.get('location', 'Unknown Location')
        return TBELocationInfo(
            estimate_project_id=estimate_project_id,
            location_name=location,
            address=location
        )
    
    @staticmethod
    def _print_final_summary(
        project_id, estimate_project_id, location_id, boq_id,
        total_items, items_with_prices, items_without_prices,
        total_estimated_supply, total_estimated_labour,
        total_estimated_amount, file_time, embedding_time,
        price_time, total_time
    ):
        """Print final processing summary."""
        print(f"\n{'='*80}")
        print("✓ COMPLETE PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Project ID:               {project_id}")
        print(f"Estimate Project ID:      {estimate_project_id}")
        print(f"Location ID:              {location_id}")
        print(f"BOQ ID:                   {boq_id}")
        print(f"\nItems:")
        print(f"  Total Items:            {total_items}")
        print(f"  Items with Prices:      {items_with_prices}")
        print(f"  Items without Prices:   {items_without_prices}")
        print(f"\nEstimated Totals:")
        print(f"  Supply Amount:          ₹{total_estimated_supply:,.2f}")
        print(f"  Labour Amount:          ₹{total_estimated_labour:,.2f}")
        print(f"  Total Amount:           ₹{total_estimated_amount:,.2f}")
        print(f"\nTiming:")
        print(f"  File Processing:        {file_time:.2f}s")
        print(f"  Embedding Generation:   {embedding_time:.2f}s")
        print(f"  Price Fetching:         {price_time:.2f}s")
        print(f"  Total Time:             {total_time:.2f}s")
        print(f"{'='*80}\n")