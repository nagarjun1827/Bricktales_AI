import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from app.core.config import settings
from app.repositories.embedding_repository import EmbeddingRepository
from app.models.embedding_dto import SimilarItemResponse

class EmbeddingService:
    def __init__(self):
        self.repo = EmbeddingRepository()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
    def initialize_database(self):
        """Initialize database with vector extension"""
        print("Initializing database for embeddings...")
        self.repo.ensure_vector_extension()
        print("Database initialized successfully")

    def generate_embeddings_for_boq(
        self,
        boq_id: int,
        model_name: str = "models/text-embedding-004",
        batch_size: int = 100,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """Generate embeddings for all items in a BOQ"""
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🧠 EMBEDDING GENERATION SERVICE")
        print(f"{'='*70}")
        print(f"BOQ ID: {boq_id}")
        print(f"Model: {model_name}")
        print(f"Batch Size: {batch_size}\n")
        
        try:
            # Get items
            if force_regenerate:
                print("📋 Fetching ALL items (force regenerate)...")
                items = self.repo.get_all_items_by_boq(boq_id)
            else:
                print("📋 Fetching items without embeddings...")
                items = self.repo.get_items_without_embeddings(boq_id)
            
            if not items:
                print("✓ All items already have embeddings\n")
                stats = self.repo.get_embedding_stats(boq_id)
                return {
                    "success": True,
                    "boq_id": boq_id,
                    "total_items": stats["total_items"],
                    "embeddings_created": 0,
                    "processing_time": time.time() - start_time,
                    "message": "No new embeddings needed"
                }
            
            print(f"✓ Found {len(items)} items to process\n")
            
            # Process in batches
            total_created = 0
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(items) + batch_size - 1) // batch_size
                
                print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
                
                # Generate embeddings
                item_ids, embeddings = self._generate_embeddings_batch(batch, model_name)
                
                # Update database
                if item_ids and embeddings:
                    updated = self.repo.update_embeddings_batch(item_ids, embeddings)
                    total_created += updated
                    print(f"   ✓ Updated {updated} embeddings")
                else:
                    print(f"   ⚠ No embeddings generated for this batch")
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*70}")
            print("✓ EMBEDDING GENERATION COMPLETE")
            print(f"{'='*70}")
            print(f"BOQ ID:              {boq_id}")
            print(f"Total Items:         {len(items)}")
            print(f"Embeddings Created:  {total_created}")
            print(f"Processing Time:     {elapsed:.2f}s")
            print(f"{'='*70}\n")
            
            return {
                "success": True,
                "boq_id": boq_id,
                "total_items": len(items),
                "embeddings_created": total_created,
                "processing_time": elapsed
            }
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_embeddings_batch(
        self,
        items: List,
        model_name: str
    ) -> tuple[List[int], List[List[float]]]:
        """Generate embeddings for a batch of items"""
        
        item_ids = []
        embeddings = []
        
        for item in items:
            try:
                # Generate embedding using Gemini
                result = genai.embed_content(
                    model=model_name,
                    content=item.item_description,
                    task_type="retrieval_document"
                )
                
                item_ids.append(item.item_id)
                embeddings.append(result['embedding'])
                
            except Exception as e:
                print(f"      ⚠ Failed to generate embedding for item {item.item_id}: {e}")
                continue
        
        return item_ids, embeddings

    def search_similar_items(
        self,
        query: str,
        top_k: int = 10,
        boq_id: Optional[int] = None,
        location_id: Optional[int] = None,
        min_similarity: float = 0.5,
        model_name: str = "models/text-embedding-004"
    ) -> Dict[str, Any]:
        """Search for similar items based on query"""
        
        try:
            print(f"\n🔍 Searching for similar items...")
            print(f"Query: {query}")
            print(f"Top K: {top_k}")
            
            # Generate embedding for query
            result = genai.embed_content(
                model=model_name,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = result['embedding']
            
            # Search database
            results = self.repo.search_similar_items(
                query_embedding=query_embedding,
                top_k=top_k,
                boq_id=boq_id,
                location_id=location_id,
                min_similarity=min_similarity
            )
            
            print(f"✓ Found {len(results)} matching items\n")
            
            # Convert to response format
            items = [
                SimilarItemResponse(
                    item_id=item.item_id,
                    item_code=item.item_code,
                    item_description=item.item_description,
                    unit_of_measurement=item.unit_of_measurement,
                    quantity=float(item.quantity),
                    supply_unit_rate=float(item.supply_unit_rate) if item.supply_unit_rate else None,
                    supply_amount=float(item.supply_amount) if item.supply_amount else None,
                    labour_unit_rate=float(item.labour_unit_rate) if item.labour_unit_rate else None,
                    labour_amount=float(item.labour_amount) if item.labour_amount else None,
                    total_amount=float(item.total_amount) if item.total_amount else None,
                    similarity_score=round(similarity, 4)
                )
                for item, similarity in results
            ]
            
            return {
                "query": query,
                "total_results": len(items),
                "items": items
            }
            
        except Exception as e:
            print(f"✗ Search error: {e}")
            raise

    def get_embedding_statistics(self, boq_id: Optional[int] = None) -> Dict[str, Any]:
        """Get embedding statistics"""
        return self.repo.get_embedding_stats(boq_id)

    def regenerate_embeddings(self, boq_id: int, model_name: str = "models/text-embedding-004") -> Dict[str, Any]:
        """Delete and regenerate all embeddings for a BOQ"""
        print(f"🔄 Regenerating embeddings for BOQ {boq_id}...")
        
        # Clear existing embeddings
        deleted = self.repo.delete_embeddings_by_boq(boq_id)
        print(f"   ✓ Cleared {deleted} existing embeddings")
        
        # Generate new embeddings
        return self.generate_embeddings_for_boq(boq_id, model_name=model_name, force_regenerate=True)