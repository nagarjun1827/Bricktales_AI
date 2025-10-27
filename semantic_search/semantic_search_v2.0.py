"""
Simple Semantic Search for BoQ Items
Search your BoQ database using natural language - Returns top 5 results
"""

import google.generativeai as genai
import psycopg2
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()


class BoQSemanticSearch:
    """Simple semantic search for BoQ items"""
    
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
        print("✓ Connected to database")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def search(self, query: str) -> List[Dict]:
        """
        Search for items using natural language - Returns top 5 results
        
        Args:
            query: Natural language search query (e.g., "concrete foundation work")
        
        Returns:
            Top 5 matching items with details and similarity scores
        """
        print(f"\n🔍 Searching for: '{query}'")
        print("Generating query embedding...")
        
        # Generate embedding for query
        query_embedding = self.generate_query_embedding(query)
        query_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        print("Searching database...")
        
        # Search database using cosine similarity - TOP 5 ONLY
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
            JOIN store_boq_files bf ON bi.file_id = bf.file_id
            JOIN projects p ON bf.project_id = p.project_id
            JOIN locations l ON bi.location_id = l.location_id
            WHERE bi.description_embedding IS NOT NULL
            ORDER BY bi.description_embedding <=> %s::vector
            LIMIT 5
        """, (query_str, query_str))
        
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
                'similarity': row[12]
            })
        
        return formatted_results
    
    def display_results(self, results: List[Dict]):
        """Display top 5 search results"""
        if not results:
            print("\n❌ No results found")
            return
        
        print(f"\n✓ Top {len(results)} results:\n")
        print("="*80)
        
        for idx, result in enumerate(results, 1):
            code_display = result['item_code'] if result['item_code'] else "N/A"
            
            print(f"\n{idx}. [{code_display}] {result['description']}")
            print(f"   Similarity: {result['similarity']:.3f} ({result['similarity']*100:.1f}%)")
            print(f"   Quantity: {result['quantity']} {result['unit']}")
            
            if result['supply_rate']:
                print(f"   Supply Rate: ₹{result['supply_rate']:,.2f}")
            if result['labour_rate']:
                print(f"   Labour Rate: ₹{result['labour_rate']:,.2f}")
            if result['total_amount']:
                print(f"   Total Amount: ₹{result['total_amount']:,.2f}")
            
            print(f"   Project: {result['project_name']} ({result['project_code']})")
            print(f"   Location: {result['location']}")
        
        print("\n" + "="*80)
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("\n✓ Database connection closed")


def main():
    """Main execution - Interactive search"""
    import sys
    
    print("="*80)
    print("BoQ Semantic Search - Top 5 Results")
    print("="*80)
    
    # Initialize searcher
    searcher = BoQSemanticSearch()
    
    # If query provided as command line argument
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        results = searcher.search(query)
        searcher.display_results(results)
        searcher.close()
        return
    
    # Interactive mode
    print("\nEnter your search queries (or 'quit' to exit)")
    print("="*80)
    
    while True:
        try:
            query = input("\n🔍 Search: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            # Perform search
            results = searcher.search(query)
            searcher.display_results(results)
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    searcher.close()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()