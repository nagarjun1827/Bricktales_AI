import google.generativeai as genai
import psycopg2
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class BoQSemanticSearch:
    """Semantic search for BoQ items using Gemini embeddings"""
    
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
            password=os.getenv('DB_PASSWORD','root'),
            port=int(os.getenv('DB_PORT', 5432))
        )
        self.cursor = self.conn.cursor()
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"  # Note: different task type for queries
        )
        return result['embedding']
    
    def search(self, query: str, limit: int = 10, min_similarity: float = 0.0) -> List[Dict]:
        """
        Search for items using natural language
        
        Args:
            query: Natural language search query (e.g., "concrete foundation work")
            limit: Maximum number of results to return
            min_similarity: Minimum similarity score (0-1)
        
        Returns:
            List of matching items with details and similarity scores
        """
        print(f"\n🔍 Searching for: '{query}'")
        print("Generating query embedding...")
        
        # Generate embedding for query
        query_embedding = self.generate_query_embedding(query)
        query_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        print("Searching database...")
        
        # Search database using cosine similarity
        self.cursor.execute("""
            SELECT 
                bi.item_id,
                bi.item_code,
                bi.item_description,
                bi.unit_of_measurement,
                bi.quantity,
                bi.supply_unit_rate,
                bi.labour_unit_rate,
                bi.total_amount,
                p.project_name,
                p.project_code,
                l.location_name,
                bf.file_name,
                1 - (bi.description_embedding <=> %s::vector) as similarity,
                bi.description_embedding <=> %s::vector as distance
            FROM store_boq_items bi
            JOIN store_boq_files bf ON bi.boq_id = bf.boq_id
            JOIN projects p ON bf.project_id = p.project_id
            JOIN locations l ON bi.location_id = l.location_id
            WHERE bi.description_embedding IS NOT NULL
              AND (1 - (bi.description_embedding <=> %s::vector)) >= %s
            ORDER BY bi.description_embedding <=> %s::vector
            LIMIT %s
        """, (query_str, query_str, query_str, min_similarity, query_str, limit))
        
        results = self.cursor.fetchall()
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'item_id': row[0],
                'item_code': row[1],
                'description': row[2],
                'unit': row[3],
                'quantity': row[4],
                'supply_rate': row[5],
                'labour_rate': row[6],
                'total_amount': row[7],
                'project_name': row[8],
                'project_code': row[9],
                'location': row[10],
                'file_name': row[11],
                'similarity': row[12],
                'distance': row[13]
            })
        
        return formatted_results
    
    def display_results(self, results: List[Dict], show_details: bool = True):
        """Display search results in a formatted way"""
        if not results:
            print("\n❌ No results found")
            return
        
        print(f"\n✓ Found {len(results)} results:\n")
        print("="*80)
        
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. [{result['item_code']}] {result['description']}")
            print(f"   Similarity: {result['similarity']:.3f} ({result['similarity']*100:.1f}%)")
            
            if show_details:
                print(f"   Quantity: {result['quantity']} {result['unit']}")
                
                if result['supply_rate']:
                    print(f"   Supply Rate: ₹{result['supply_rate']:,.2f}")
                if result['labour_rate']:
                    print(f"   Labour Rate: ₹{result['labour_rate']:,.2f}")
                if result['total_amount']:
                    print(f"   Total Amount: ₹{result['total_amount']:,.2f}")
                
                print(f"   Project: {result['project_name']} ({result['project_code']})")
                print(f"   Location: {result['location']}")
                print(f"   File: {result['file_name']}")
        
        print("\n" + "="*80)
    
    def find_similar_to_item(self, item_id: int, limit: int = 10) -> List[Dict]:
        """Find items similar to a specific item"""
        print(f"\n🔍 Finding items similar to item ID: {item_id}")
        
        # Get the reference item
        self.cursor.execute("""
            SELECT item_code, item_description, description_embedding
            FROM store_boq_items
            WHERE item_id = %s AND description_embedding IS NOT NULL
        """, (item_id,))
        
        ref_item = self.cursor.fetchone()
        if not ref_item:
            print(f"❌ Item {item_id} not found or has no embedding")
            return []
        
        ref_code, ref_desc, ref_embedding = ref_item
        print(f"Reference: [{ref_code}] {ref_desc[:60]}...")
        
        # Convert embedding to string format
        embedding_str = '[' + ','.join(map(str, ref_embedding)) + ']'
        
        # Search for similar items
        self.cursor.execute("""
            SELECT 
                bi.item_id,
                bi.item_code,
                bi.item_description,
                bi.unit_of_measurement,
                bi.quantity,
                bi.supply_unit_rate,
                bi.labour_unit_rate,
                bi.total_amount,
                p.project_name,
                p.project_code,
                l.location_name,
                bf.file_name,
                1 - (bi.description_embedding <=> %s::vector) as similarity
            FROM store_boq_items bi
            JOIN store_boq_files bf ON bi.boq_id = bf.boq_id
            JOIN projects p ON bf.project_id = p.project_id
            JOIN locations l ON bi.location_id = l.location_id
            WHERE bi.description_embedding IS NOT NULL
              AND bi.item_id != %s
            ORDER BY bi.description_embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, item_id, embedding_str, limit))
        
        results = self.cursor.fetchall()
        
        formatted_results = []
        for row in results:
            formatted_results.append({
                'item_id': row[0],
                'item_code': row[1],
                'description': row[2],
                'unit': row[3],
                'quantity': row[4],
                'supply_rate': row[5],
                'labour_rate': row[6],
                'total_amount': row[7],
                'project_name': row[8],
                'project_code': row[9],
                'location': row[10],
                'file_name': row[11],
                'similarity': row[12]
            })
        
        return formatted_results
    
    def estimate_cost(self, query: str, top_k: int = 10) -> Dict:
        """
        Estimate cost for a work item based on similar items
        
        Args:
            query: Description of the work item
            top_k: Number of similar items to consider
        
        Returns:
            Dictionary with cost statistics
        """
        print(f"\n💰 Estimating cost for: '{query}'")
        
        # Search for similar items
        results = self.search(query, limit=top_k, min_similarity=0.5)
        
        if not results:
            print("❌ No similar items found for cost estimation")
            return None
        
        # Extract rates from items with the same unit
        rates_by_unit = {}
        
        for result in results:
            unit = result['unit']
            rate = result['supply_rate']
            
            if rate and rate > 0:
                if unit not in rates_by_unit:
                    rates_by_unit[unit] = []
                rates_by_unit[unit].append({
                    'rate': rate,
                    'description': result['description'],
                    'project': result['project_name'],
                    'similarity': result['similarity']
                })
        
        if not rates_by_unit:
            print("❌ No items with rates found")
            return None
        
        # Calculate statistics for each unit
        print("\n📊 Cost Estimation Results:\n")
        
        for unit, items in rates_by_unit.items():
            rates = [item['rate'] for item in items]
            
            print(f"Unit: {unit}")
            print(f"  Sample Size: {len(rates)} items")
            print(f"  Average Rate: ₹{sum(rates)/len(rates):,.2f}")
            print(f"  Median Rate: ₹{sorted(rates)[len(rates)//2]:,.2f}")
            print(f"  Min Rate: ₹{min(rates):,.2f}")
            print(f"  Max Rate: ₹{max(rates):,.2f}")
            print(f"  Range: ₹{max(rates) - min(rates):,.2f}")
            print(f"\n  Based on:")
            for item in items[:3]:  # Show top 3
                print(f"    • ₹{item['rate']:,.2f} - {item['description'][:50]}...")
                print(f"      (Similarity: {item['similarity']:.3f}, Project: {item['project']})")
            print()
        
        return rates_by_unit
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def interactive_search():
    """Interactive search mode"""
    print("="*80)
    print("BoQ Semantic Search - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  search <query>     - Search for items")
    print("  similar <item_id>  - Find items similar to item ID")
    print("  estimate <query>   - Estimate cost for work item")
    print("  quit              - Exit")
    print("="*80)
    
    searcher = BoQSemanticSearch()
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            
            if cmd == 'search' and len(parts) > 1:
                query = parts[1]
                results = searcher.search(query, limit=10)
                searcher.display_results(results)
            
            elif cmd == 'similar' and len(parts) > 1:
                try:
                    item_id = int(parts[1])
                    results = searcher.find_similar_to_item(item_id, limit=10)
                    searcher.display_results(results)
                except ValueError:
                    print("❌ Invalid item ID. Please provide a number.")
            
            elif cmd == 'estimate' and len(parts) > 1:
                query = parts[1]
                searcher.estimate_cost(query, top_k=15)
            
            else:
                print("❌ Unknown command. Use: search, similar, estimate, or quit")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    searcher.close()
    print("\n👋 Goodbye!")


def main():
    """Main execution with example searches"""
    import sys
    
    # If arguments provided, do a quick search
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        
        searcher = BoQSemanticSearch()
        results = searcher.search(query, limit=5)
        searcher.display_results(results)
        searcher.close()
        return
    
    # Otherwise, run interactive mode
    interactive_search()


if __name__ == "__main__":
    main()