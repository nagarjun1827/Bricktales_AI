"""
Gemini Vectorizer - HIGH PERFORMANCE VERSION
Features:
- Parallel processing with ThreadPoolExecutor
- Batch embedding requests
- Automatic rate limit handling
- Progress tracking
- Error recovery
"""

import google.generativeai as genai
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Optional, Tuple
import os
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

load_dotenv()


class GeminiVectorizerParallel:
    """High-performance embedding generator using parallel processing"""
    
    def __init__(self, max_workers: int = 5, batch_size: int = 10):
        """
        Initialize vectorizer
        
        Args:
            max_workers: Number of parallel workers (default: 5)
            batch_size: Number of items to process before committing (default: 10)
        """
        print("="*70)
        print("Gemini Vectorizer - HIGH PERFORMANCE MODE")
        print("="*70)
        
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model_name = 'models/embedding-001'
        self.embedding_dim = 768
        
        print(f"✓ Gemini configured")
        print(f"  Model: {self.model_name}")
        print(f"  Dimensions: {self.embedding_dim}")
        print(f"  Max Workers: {max_workers}")
        print(f"  Batch Size: {batch_size}")
        
        # Connect to database
        try:
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'database': os.getenv('DB_NAME', 'boq_database'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'root'),
                'port': int(os.getenv('DB_PORT', 5432))
            }
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("✓ Database connected")
        except Exception as e:
            raise Exception(f"❌ Database connection failed: {e}")
        
        # Thread-safe counters
        self.lock = threading.Lock()
        self.processed = 0
        self.errors = 0
        self.rate_limit_hits = 0
    
    def check_setup(self):
        """Verify database setup"""
        print("\nChecking database setup...")
        
        # Check pgvector extension
        self.cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        if not self.cursor.fetchone():
            raise Exception("❌ pgvector extension not found. Run: CREATE EXTENSION vector;")
        print("✓ pgvector extension found")
        
        # Check if vector column exists
        self.cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'store_boq_items' 
            AND column_name = 'description_embedding'
        """)
        result = self.cursor.fetchone()
        if not result:
            raise Exception("❌ description_embedding column not found in store_boq_items table")
        print(f"✓ Vector column found: {result[0]} ({result[1]})")
        
        # Check if index exists
        self.cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'store_boq_items' 
            AND indexname LIKE '%embedding%'
        """)
        indexes = self.cursor.fetchall()
        if indexes:
            print(f"✓ Vector index found: {indexes[0][0]}")
        else:
            print("⚠️  No vector index found (will be slower for searches)")
        
        # Count items to process
        self.cursor.execute("""
            SELECT COUNT(*) FROM store_boq_items WHERE description_embedding IS NULL
        """)
        count = self.cursor.fetchone()[0]
        print(f"✓ Found {count} items without embeddings")
        
        return count
    
    def generate_embedding_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Generate embedding with automatic retry on rate limits"""
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limits
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    with self.lock:
                        self.rate_limit_hits += 1
                    
                    wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                    print(f"  ⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Other errors
                else:
                    print(f"  ❌ Embedding error: {e}")
                    return None
        
        return None
    
    def process_item(self, item: Tuple) -> Optional[Tuple]:
        """Process a single item - generate embedding"""
        item_id, code, desc, unit, qty = item
        
        # Create rich text representation
        code_text = code if code else "No code"
        text = f"Item {code_text}: {desc}. Quantity: {qty} {unit}"
        
        # Generate embedding
        embedding = self.generate_embedding_with_retry(text)
        
        if embedding is None:
            with self.lock:
                self.errors += 1
            return None
        
        # Convert to PostgreSQL array format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        with self.lock:
            self.processed += 1
        
        return (item_id, embedding_str)
    
    def batch_update_embeddings(self, results: List[Tuple]):
        """Batch update embeddings in database"""
        if not results:
            return
        
        # Create new connection for this thread (psycopg2 connections aren't thread-safe)
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Simpler approach: Update each item individually within a transaction
            # This is still efficient because it's within one transaction
            for item_id, embedding_str in results:
                cursor.execute("""
                    UPDATE store_boq_items 
                    SET description_embedding = %s::vector,
                        embedding_model = %s,
                        embedding_generated_at = NOW()
                    WHERE item_id = %s
                """, (embedding_str, self.model_name, item_id))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"  ❌ Batch update error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cursor.close()
            conn.close()
    
    def vectorize_parallel(self, file_id: Optional[int] = None, skip_existing: bool = True):
        """Generate embeddings using parallel processing"""
        print("\n" + "="*70)
        print("PARALLEL EMBEDDING GENERATION")
        print("="*70)
        
        # Build query
        where_conditions = []
        params = []
        
        if skip_existing:
            where_conditions.append("description_embedding IS NULL")
        
        if file_id:
            where_conditions.append("file_id = %s")
            params.append(file_id)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
            SELECT item_id, item_code, item_description, 
                   unit_of_measurement, quantity
            FROM store_boq_items 
            {where_clause}
            ORDER BY item_id
        """
        
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        
        items = self.cursor.fetchall()
        total = len(items)
        
        if total == 0:
            print("✓ No items to process (all items already have embeddings)")
            return
        
        print(f"\nProcessing {total} items with {self.max_workers} parallel workers...")
        print(f"Batch size: {self.batch_size} items")
        print()
        
        start_time = time.time()
        self.processed = 0
        self.errors = 0
        self.rate_limit_hits = 0
        
        # Process items in parallel
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all items
            future_to_item = {executor.submit(self.process_item, item): item for item in items}
            
            # Process results as they complete
            for idx, future in enumerate(as_completed(future_to_item), 1):
                try:
                    result = future.result()
                    
                    if result:
                        batch_results.append(result)
                    
                    # Batch update when batch_size is reached
                    if len(batch_results) >= self.batch_size:
                        self.batch_update_embeddings(batch_results)
                        batch_results = []
                    
                    # Progress update
                    if idx % 10 == 0 or idx == total:
                        elapsed = time.time() - start_time
                        rate = self.processed / elapsed if elapsed > 0 else 0
                        progress_pct = (idx * 100) // total
                        
                        print(f"  Progress: {idx}/{total} ({progress_pct}%) | "
                              f"Success: {self.processed} | "
                              f"Errors: {self.errors} | "
                              f"Rate Limits: {self.rate_limit_hits} | "
                              f"Speed: {rate:.2f} items/s")
                
                except Exception as e:
                    print(f"  ❌ Future error: {e}")
                    with self.lock:
                        self.errors += 1
        
        # Update remaining batch
        if batch_results:
            self.batch_update_embeddings(batch_results)
        
        elapsed_total = time.time() - start_time
        
        print("\n" + "="*70)
        print("✓ PARALLEL EMBEDDING GENERATION COMPLETE")
        print("="*70)
        print(f"Total items: {total}")
        print(f"Successfully processed: {self.processed}")
        print(f"Errors: {self.errors}")
        print(f"Rate limit hits: {self.rate_limit_hits}")
        print(f"Total time: {elapsed_total:.2f} seconds")
        print(f"Average speed: {self.processed/elapsed_total:.2f} items/second")
        print(f"Success rate: {(self.processed/total)*100:.1f}%")
        print("="*70)
    
    def verify_embeddings(self):
        """Verify that embeddings were created"""
        print("\nVerifying embeddings...")
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM store_boq_items WHERE description_embedding IS NOT NULL
        """)
        with_embeddings = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM store_boq_items")
        total_items = self.cursor.fetchone()[0]
        
        print(f"✓ Items with embeddings: {with_embeddings}/{total_items} ({(with_embeddings/total_items)*100:.1f}%)")
        
        if with_embeddings > 0:
            # Test similarity search
            print("\nTesting similarity search...")
            self.cursor.execute("""
                SELECT item_code, item_description,
                       description_embedding <=> (
                           SELECT description_embedding 
                           FROM store_boq_items 
                           WHERE description_embedding IS NOT NULL 
                           LIMIT 1
                       ) as distance
                FROM store_boq_items
                WHERE description_embedding IS NOT NULL
                ORDER BY distance
                LIMIT 5
            """)
            results = self.cursor.fetchall()
            
            print("\nTop 5 similar items:")
            for idx, (code, desc, dist) in enumerate(results, 1):
                code_display = code if code else "N/A"
                desc_short = desc[:60] + "..." if len(desc) > 60 else desc
                print(f"  {idx}. [{code_display}] {desc_short}")
                print(f"     Distance: {dist:.4f}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("\n✓ Database connection closed")


def main():
    """Main execution"""
    import sys
    
    # Configuration
    max_workers = 5      # Number of parallel threads
    batch_size = 10      # Number of items to batch before DB update
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_workers: {sys.argv[1]}, using default: 5")
    
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            print(f"Invalid batch_size: {sys.argv[2]}, using default: 10")
    
    try:
        # Initialize
        vectorizer = GeminiVectorizerParallel(
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        # Check setup
        items_to_process = vectorizer.check_setup()
        
        if items_to_process == 0:
            print("\n✓ All items already have embeddings!")
            vectorizer.close()
            return
        
        # Estimate time
        # Gemini rate limit: ~60 req/min = 1 req/sec
        # With 5 workers and rate limiting, expect ~3-4 items/sec
        estimated_time = items_to_process / 3.5 / 60  # minutes
        
        print(f"\n" + "="*70)
        print(f"Ready to process {items_to_process} items")
        print(f"Workers: {max_workers} parallel threads")
        print(f"Batch size: {batch_size} items")
        print(f"Estimated time: ~{estimated_time:.1f} minutes")
        print("="*70)
        
        response = input("\nProceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled by user")
            vectorizer.close()
            return
        
        # Generate embeddings
        vectorizer.vectorize_parallel()
        
        # Verify
        vectorizer.verify_embeddings()
        
        # Close
        vectorizer.close()
        
        print("\n" + "="*70)
        print("✅ SUCCESS! Parallel embedding generation complete")
        print("="*70)
        print("\nYou can now use semantic search on your BoQ items!")
        print("\nExample similarity search query:")
        print("""
        SELECT 
            item_code, 
            item_description,
            description_embedding <=> '[your_query_embedding]'::vector AS distance
        FROM store_boq_items
        WHERE description_embedding IS NOT NULL
        ORDER BY distance
        LIMIT 10;
        """)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Partial progress has been saved to database")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()