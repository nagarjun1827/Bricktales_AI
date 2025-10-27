import google.generativeai as genai
import psycopg2
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import json

load_dotenv()


class BoQPriceFetcher:
    """Fetch prices for BoQ line items using semantic search"""
    
    def __init__(self):
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model_name = 'models/embedding-001'
        
        # Connect to database
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'boq_database'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'root'),
            port=int(os.getenv('DB_PORT', 5432))
        )
        self.cursor = self.conn.cursor()
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def find_similar_store_items(self, description: str, unit: str, limit: int = 5, min_similarity: float = 0.5) -> List[Dict]:
        """
        Find similar items from store_boq_items
        
        Args:
            description: Item description to search for
            unit: Unit of measurement
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar items with prices
        """
        # Generate embedding for the description
        query_embedding = self.generate_query_embedding(description)
        query_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Search store_boq_items
        self.cursor.execute("""
            SELECT 
                bi.item_id,
                bi.item_code,
                bi.item_description,
                bi.unit_of_measurement,
                bi.supply_unit_rate,
                bi.labour_unit_rate,
                bi.total_amount,
                p.project_name,
                bf.file_name,
                1 - (bi.description_embedding <=> %s::vector) as similarity
            FROM store_boq_items bi
            JOIN store_boq_files bf ON bi.boq_id = bf.boq_id
            JOIN projects p ON bf.project_id = p.project_id
            WHERE bi.description_embedding IS NOT NULL
              AND bi.unit_of_measurement = %s
              AND (1 - (bi.description_embedding <=> %s::vector)) >= %s
              AND bi.supply_unit_rate IS NOT NULL
              AND bi.supply_unit_rate > 0
            ORDER BY bi.description_embedding <=> %s::vector
            LIMIT %s
        """, (query_str, unit, query_str, min_similarity, query_str, limit))
        
        results = self.cursor.fetchall()
        
        formatted_results = []
        for row in results:
            formatted_results.append({
                'item_id': row[0],
                'item_code': row[1],
                'description': row[2],
                'unit': row[3],
                'supply_rate': row[4],
                'labour_rate': row[5],
                'total_amount': row[6],
                'project_name': row[7],
                'file_name': row[8],
                'similarity': row[9]
            })
        
        return formatted_results
    
    def calculate_price_statistics(self, similar_items: List[Dict]) -> Dict:
        """Calculate price statistics from similar items"""
        if not similar_items:
            return None
        
        supply_rates = [item['supply_rate'] for item in similar_items if item['supply_rate']]
        labour_rates = [item['labour_rate'] for item in similar_items if item['labour_rate']]
        
        stats = {}
        
        if supply_rates:
            stats['supply_rate'] = {
                'avg': sum(supply_rates) / len(supply_rates),
                'min': min(supply_rates),
                'max': max(supply_rates),
                'median': sorted(supply_rates)[len(supply_rates) // 2],
                'count': len(supply_rates)
            }
        
        if labour_rates:
            stats['labour_rate'] = {
                'avg': sum(labour_rates) / len(labour_rates),
                'min': min(labour_rates),
                'max': max(labour_rates),
                'median': sorted(labour_rates)[len(labour_rates) // 2],
                'count': len(labour_rates)
            }
        
        return stats
    
    def fetch_prices_for_boq(self, boq_id: int, top_k: int = 5, min_similarity: float = 0.5) -> List[Dict]:
        """
        Fetch prices for all items in a BoQ
        
        Args:
            boq_id: BoQ ID to fetch prices for
            top_k: Number of similar items to consider per item
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of items with price recommendations
        """
        print(f"\n📋 Fetching prices for BoQ ID: {boq_id}")
        print("="*80)
        
        # Get all items for this BoQ
        self.cursor.execute("""
            SELECT 
                item_id,
                item_code,
                item_description,
                unit_of_measurement,
                quantity
            FROM store_boq_items
            WHERE boq_id = %s
            ORDER BY item_id
        """, (boq_id,))
        
        line_items = self.cursor.fetchall()
        
        if not line_items:
            print(f"❌ No line items found for BoQ ID: {boq_id}")
            return []
        
        print(f"\n✓ Found {len(line_items)} line items")
        print("Searching for prices...\n")
        
        results = []
        
        for idx, (line_item_id, item_code, description, unit, quantity) in enumerate(line_items, 1):
            print(f"[{idx}/{len(line_items)}] Processing: [{item_code}] {description[:50]}...")
            
            # Find similar items from store
            similar_items = self.find_similar_store_items(
                description=description,
                unit=unit,
                limit=top_k,
                min_similarity=min_similarity
            )
            
            if not similar_items:
                print(f"  ⚠️  No similar items found (similarity < {min_similarity})")
                results.append({
                    'line_item_id': line_item_id,
                    'item_code': item_code,
                    'description': description,
                    'unit': unit,
                    'quantity': quantity,
                    'similar_items_found': 0,
                    'price_stats': None,
                    'similar_items': []
                })
                continue
            
            # Calculate price statistics
            price_stats = self.calculate_price_statistics(similar_items)
            
            print(f"  ✓ Found {len(similar_items)} similar items")
            if price_stats and 'supply_rate' in price_stats:
                avg_rate = price_stats['supply_rate']['avg']
                print(f"  💰 Avg Supply Rate: ₹{avg_rate:,.2f}/{unit}")
            
            results.append({
                'line_item_id': line_item_id,
                'item_code': item_code,
                'description': description,
                'unit': unit,
                'quantity': quantity,
                'similar_items_found': len(similar_items),
                'price_stats': price_stats,
                'similar_items': similar_items
            })
        
        return results
    
    def display_results(self, results: List[Dict], show_similar_items: bool = False):
        """Display price fetching results"""
        if not results:
            print("\n❌ No results to display")
            return
        
        print("\n" + "="*80)
        print("PRICE RECOMMENDATIONS")
        print("="*80)
        
        total_items = len(results)
        items_with_prices = sum(1 for r in results if r['price_stats'])
        
        print(f"\nTotal Line Items: {total_items}")
        print(f"Items with Price Recommendations: {items_with_prices}")
        print(f"Items without Recommendations: {total_items - items_with_prices}")
        
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. [{result['item_code']}] {result['description']}")
            print(f"   Quantity: {result['quantity']} {result['unit']}")
            
            if result['price_stats']:
                stats = result['price_stats']
                
                if 'supply_rate' in stats:
                    sr = stats['supply_rate']
                    print(f"\n   📊 Supply Rate Statistics (based on {sr['count']} similar items):")
                    print(f"      Average: ₹{sr['avg']:,.2f}")
                    print(f"      Median:  ₹{sr['median']:,.2f}")
                    print(f"      Range:   ₹{sr['min']:,.2f} - ₹{sr['max']:,.2f}")
                    print(f"      Estimated Total: ₹{sr['avg'] * result['quantity']:,.2f}")
                
                if 'labour_rate' in stats:
                    lr = stats['labour_rate']
                    print(f"\n   📊 Labour Rate Statistics (based on {lr['count']} similar items):")
                    print(f"      Average: ₹{lr['avg']:,.2f}")
                    print(f"      Median:  ₹{lr['median']:,.2f}")
                    print(f"      Range:   ₹{lr['min']:,.2f} - ₹{lr['max']:,.2f}")
                
                if show_similar_items and result['similar_items']:
                    print(f"\n   🔍 Similar Items Found:")
                    for i, item in enumerate(result['similar_items'][:3], 1):
                        print(f"      {i}. {item['description'][:50]}...")
                        print(f"         Rate: ₹{item['supply_rate']:,.2f} | Similarity: {item['similarity']:.3f}")
                        print(f"         Project: {item['project_name']}")
            else:
                print(f"   ⚠️  No similar items found (similarity < threshold)")
            
            print()
        
        print("="*80)
    
    def export_to_csv(self, results: List[Dict], output_file: str = "price_recommendations.csv"):
        """Export results to CSV file"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Line Item ID', 'Item Code', 'Description', 'Unit', 'Quantity',
                'Similar Items Found', 
                'Avg Supply Rate', 'Min Supply Rate', 'Max Supply Rate',
                'Avg Labour Rate', 'Min Labour Rate', 'Max Labour Rate',
                'Estimated Total (Supply)'
            ])
            
            # Write data
            for result in results:
                stats = result['price_stats']
                
                avg_supply = stats['supply_rate']['avg'] if stats and 'supply_rate' in stats else None
                min_supply = stats['supply_rate']['min'] if stats and 'supply_rate' in stats else None
                max_supply = stats['supply_rate']['max'] if stats and 'supply_rate' in stats else None
                
                avg_labour = stats['labour_rate']['avg'] if stats and 'labour_rate' in stats else None
                min_labour = stats['labour_rate']['min'] if stats and 'labour_rate' in stats else None
                max_labour = stats['labour_rate']['max'] if stats and 'labour_rate' in stats else None
                
                estimated_total = avg_supply * result['quantity'] if avg_supply else None
                
                writer.writerow([
                    result['line_item_id'],
                    result['item_code'],
                    result['description'],
                    result['unit'],
                    result['quantity'],
                    result['similar_items_found'],
                    f"{avg_supply:.2f}" if avg_supply else "",
                    f"{min_supply:.2f}" if min_supply else "",
                    f"{max_supply:.2f}" if max_supply else "",
                    f"{avg_labour:.2f}" if avg_labour else "",
                    f"{min_labour:.2f}" if min_labour else "",
                    f"{max_labour:.2f}" if max_labour else "",
                    f"{estimated_total:.2f}" if estimated_total else ""
                ])
        
        print(f"\n✓ Results exported to: {output_file}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_boq_prices.py <boq_id> [top_k] [min_similarity]")
        print("\nExample:")
        print("  python fetch_boq_prices.py 123")
        print("  python fetch_boq_prices.py 123 10 0.6")
        print("\nArguments:")
        print("  boq_id          - BoQ ID to fetch prices for (required)")
        print("  top_k           - Number of similar items to consider (default: 5)")
        print("  min_similarity  - Minimum similarity threshold 0-1 (default: 0.5)")
        sys.exit(1)
    
    boq_id = int(sys.argv[1])
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    min_similarity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    try:
        fetcher = BoQPriceFetcher()
        
        # Fetch prices
        results = fetcher.fetch_prices_for_boq(
            boq_id=boq_id,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        # Display results
        fetcher.display_results(results, show_similar_items=True)
        
        # Export to CSV
        csv_filename = f"price_recommendations_boq_{boq_id}.csv"
        fetcher.export_to_csv(results, csv_filename)
        
        fetcher.close()
        
        print("\n✅ Complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()