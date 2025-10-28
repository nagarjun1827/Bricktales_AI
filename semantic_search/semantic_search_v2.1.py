import google.generativeai as genai
import psycopg2
from typing import List, Dict
import os
from dotenv import load_dotenv

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
    
    def search(self, query: str, limit: int = 5, min_similarity: float = 0.0) -> List[Dict]:
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
                # print(f"   Quantity: {result['quantity']} {result['unit']}")
                
                if result['supply_rate']:
                    print(f"   Supply Rate: ₹{result['supply_rate']:,.2f}")
                if result['labour_rate']:
                    print(f"   Labour Rate: ₹{result['labour_rate']:,.2f}")
                # if result['total_amount']:
                #     print(f"   Total Amount: ₹{result['total_amount']:,.2f}")
                
                # print(f"   Project: {result['project_name']} ({result['project_code']})")
                print(f"   Location: {result['location']}")
                print(f"   File: {result['file_name']}")
        
        print("\n" + "="*80)
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def main():
    """Main execution - simple search interface"""
    searcher = BoQSemanticSearch()
    
    print("="*80)
    print("BoQ Semantic Search")
    print("="*80)
    print("\nType your search query (or 'quit' to exit)")
    
    while True:
        try:
            query = input("\n🔍 Search: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            # Perform search
            results = searcher.search(query, limit=5)
            
            # Display results
            searcher.display_results(results, show_details=True)
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    searcher.close()
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()