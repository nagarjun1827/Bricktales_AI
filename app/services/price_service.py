import time
from typing import List, Dict, Any, Optional
from decimal import Decimal
import google.generativeai as genai
from statistics import median

from app.core.config import settings
from app.repositories.price_repository import PriceRepository
from app.models.price_dto import (
    PriceStatistics,
    SimilarItemDetail,
    LineItemPriceRecommendation
)

class PriceService:
    """Service for fetching prices for BOQ line items"""
    
    def __init__(self):
        self.repo = PriceRepository()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = 'models/text-embedding-004'
    
    def fetch_prices_for_boq(
        self,
        boq_id: int,
        top_k: int = 1,
        min_similarity: float = 0.5
    ) -> Dict[str, Any]:
        """Fetch price recommendations for all items in a BOQ"""
        
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"📋 PRICE FETCHING SERVICE")
        print(f"{'='*80}")
        print(f"BOQ ID: {boq_id}")
        print(f"Top K: {top_k}")
        print(f"Min Similarity: {min_similarity}\n")
        
        try:
            # Verify BOQ exists
            boq_info = self.repo.get_boq_info(boq_id)
            if not boq_info:
                return {
                    'success': False,
                    'error': f'BOQ ID {boq_id} not found'
                }
            
            print(f"✓ BOQ: {boq_info['file_name']}")
            print(f"✓ Project: {boq_info['project_name']}\n")
            
            # Get line items
            print("📋 Fetching line items...")
            line_items = self.repo.get_boq_line_items(boq_id)
            
            if not line_items:
                return {
                    'success': False,
                    'error': 'No line items found in BOQ'
                }
            
            print(f"✓ Found {len(line_items)} line items\n")
            
            # Process each line item
            recommendations = []
            items_with_prices = 0
            
            print("🔍 Searching for similar items and prices...\n")
            
            for idx, (item_id, item_code, description, unit, quantity) in enumerate(line_items, 1):
                print(f"[{idx}/{len(line_items)}] {item_code or 'N/A'}: {description[:60]}...")
                
                # Convert quantity to float for calculations
                quantity_float = float(quantity) if isinstance(quantity, Decimal) else quantity
                
                # Generate embedding for description
                query_embedding = self._generate_embedding(description)
                
                # Find similar items
                similar_items = self.repo.find_similar_items(
                    query_embedding=query_embedding,
                    unit=unit,
                    limit=top_k,
                    min_similarity=min_similarity
                )
                
                if similar_items:
                    print(f"  ✓ Found {len(similar_items)} similar items")
                    items_with_prices += 1
                    
                    # Calculate statistics
                    supply_stats = self._calculate_price_stats(
                        [item['supply_rate'] for item in similar_items if item['supply_rate'] > 0]
                    )
                    
                    labour_stats = self._calculate_price_stats(
                        [item['labour_rate'] for item in similar_items if item['labour_rate'] > 0]
                    )
                    
                    if supply_stats:
                        print(f"  💰 Avg Supply Rate: ₹{supply_stats.avg:,.2f}/{unit}")
                    
                    # Format similar items
                    similar_details = [
                        SimilarItemDetail(
                            item_id=item['item_id'],
                            item_code=item['item_code'],
                            description=item['description'],
                            supply_rate=item['supply_rate'],
                            labour_rate=item['labour_rate'],
                            similarity=round(item['similarity'], 4),
                            project_name=item['project_name'],
                            file_name=item['file_name']
                        )
                        for item in similar_items
                    ]
                else:
                    print(f"  ⚠️  No similar items found")
                    supply_stats = None
                    labour_stats = None
                    similar_details = []
                
                # Calculate estimated totals (using float for calculations)
                estimated_supply = supply_stats.avg * quantity_float if supply_stats else None
                estimated_labour = labour_stats.avg * quantity_float if labour_stats else None
                estimated_total = None
                if estimated_supply is not None and estimated_labour is not None:
                    estimated_total = estimated_supply + estimated_labour
                elif estimated_supply is not None:
                    estimated_total = estimated_supply
                
                recommendation = LineItemPriceRecommendation(
                    line_item_id=item_id,
                    item_code=item_code,
                    description=description,
                    unit=unit,
                    quantity=quantity_float,  # Convert to float for response
                    similar_items_found=len(similar_items),
                    supply_rate_stats=supply_stats,
                    labour_rate_stats=labour_stats,
                    estimated_supply_total=estimated_supply,
                    estimated_labour_total=estimated_labour,
                    estimated_total=estimated_total,
                    similar_items=similar_details
                )
                
                recommendations.append(recommendation)
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("✓ PRICE FETCHING COMPLETE")
            print(f"{'='*80}")
            print(f"Total Items:              {len(line_items)}")
            print(f"Items with Prices:        {items_with_prices}")
            print(f"Items without Prices:     {len(line_items) - items_with_prices}")
            print(f"Processing Time:          {elapsed:.2f}s")
            print(f"{'='*80}\n")
            
            return {
                'success': True,
                'boq_id': boq_id,
                'total_items': len(line_items),
                'items_with_recommendations': items_with_prices,
                'items_without_recommendations': len(line_items) - items_with_prices,
                'processing_time': elapsed,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def _calculate_price_stats(self, rates: List[float]) -> Optional[PriceStatistics]:
        """Calculate statistics for a list of rates"""
        if not rates:
            return None
        
        # Ensure all rates are floats (in case any Decimals slip through)
        rates_float = [float(r) if isinstance(r, Decimal) else r for r in rates]
        
        return PriceStatistics(
            avg=sum(rates_float) / len(rates_float),
            min=min(rates_float),
            max=max(rates_float),
            median=median(rates_float),
            count=len(rates_float)
        )