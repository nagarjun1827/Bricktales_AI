import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from app.core.config import settings
from app.models.embedding_domain import BOQItemWithEmbedding

class EmbeddingRepository:
    def __init__(self):
        self.db_config = {
            "host": settings.DB_HOST,
            "database": settings.DB_NAME,
            "user": settings.DB_USER,
            "password": settings.DB_PASSWORD,
            "port": settings.DB_PORT,
        }

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def ensure_vector_extension(self):
        """Ensure pgvector extension is installed"""
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

    def get_items_without_embeddings(self, boq_id: int) -> List[BOQItemWithEmbedding]:
        """Get all items from a BOQ that don't have embeddings"""
        query = """
            SELECT 
                item_id, boq_id, item_code, item_description,
                unit_of_measurement, quantity, supply_unit_rate,
                supply_amount, labour_unit_rate, labour_amount,
                total_amount, location_id, created_by, created_at,
                updated_by, updated_at, description_embedding,
                embedding_generated_at
            FROM store_boq_items
            WHERE boq_id = %s AND description_embedding IS NULL
            ORDER BY item_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            rows = cur.fetchall()
            
            return [self._row_to_item(row) for row in rows]

    def get_all_items_by_boq(self, boq_id: int) -> List[BOQItemWithEmbedding]:
        """Get all items from a BOQ (with or without embeddings)"""
        query = """
            SELECT 
                item_id, boq_id, item_code, item_description,
                unit_of_measurement, quantity, supply_unit_rate,
                supply_amount, labour_unit_rate, labour_amount,
                total_amount, location_id, created_by, created_at,
                updated_by, updated_at, description_embedding,
                embedding_generated_at
            FROM store_boq_items
            WHERE boq_id = %s
            ORDER BY item_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            rows = cur.fetchall()
            
            return [self._row_to_item(row) for row in rows]

    def update_embeddings_batch(self, item_ids: List[int], embeddings: List[List[float]]) -> int:
        """Update embeddings for multiple items"""
        if not item_ids or not embeddings or len(item_ids) != len(embeddings):
            return 0
            
        query = """
            UPDATE store_boq_items
            SET description_embedding = data.embedding::vector,
                embedding_generated_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            FROM (VALUES %s) AS data(item_id, embedding)
            WHERE store_boq_items.item_id = data.item_id::integer
        """
        
        values = [
            (item_id, f"[{','.join(map(str, embedding))}]")
            for item_id, embedding in zip(item_ids, embeddings)
        ]
        
        with self._get_connection() as conn, conn.cursor() as cur:
            execute_values(cur, query, values, template="(%s, %s)")
            affected = cur.rowcount
            conn.commit()
            return affected

    def search_similar_items(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        boq_id: Optional[int] = None,
        location_id: Optional[int] = None,
        min_similarity: float = 0.5
    ) -> List[Tuple[BOQItemWithEmbedding, float]]:
        """Search for similar items using cosine similarity"""
        
        # Build WHERE clause
        where_clauses = ["description_embedding IS NOT NULL"]
        where_clauses.append("1 - (description_embedding <=> %s::vector) >= %s")
        params = [f"[{','.join(map(str, query_embedding))}]", min_similarity]
        
        if boq_id is not None:
            where_clauses.append("boq_id = %s")
            params.append(boq_id)
        
        if location_id is not None:
            where_clauses.append("location_id = %s")
            params.append(location_id)
        
        params.append(top_k)
        
        query = f"""
            SELECT 
                item_id, boq_id, item_code, item_description,
                unit_of_measurement, quantity, supply_unit_rate,
                supply_amount, labour_unit_rate, labour_amount,
                total_amount, location_id, created_by, created_at,
                updated_by, updated_at, description_embedding,
                embedding_generated_at,
                1 - (description_embedding <=> %s::vector) as similarity
            FROM store_boq_items
            WHERE {' AND '.join(where_clauses)}
            ORDER BY similarity DESC
            LIMIT %s
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                item = self._row_to_item(row[:-1])  # Exclude similarity score
                similarity = float(row[-1])
                results.append((item, similarity))
            
            return results

    def get_embedding_stats(self, boq_id: Optional[int] = None) -> Dict[str, int]:
        """Get statistics about embeddings"""
        if boq_id:
            query = """
                SELECT 
                    COUNT(*) as total_items,
                    COUNT(description_embedding) as items_with_embeddings
                FROM store_boq_items
                WHERE boq_id = %s
            """
            params = (boq_id,)
        else:
            query = """
                SELECT 
                    COUNT(*) as total_items,
                    COUNT(description_embedding) as items_with_embeddings
                FROM store_boq_items
            """
            params = None
        
        with self._get_connection() as conn, conn.cursor() as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            result = cur.fetchone()
            
            return {
                "total_items": result[0],
                "items_with_embeddings": result[1],
                "items_without_embeddings": result[0] - result[1]
            }

    def delete_embeddings_by_boq(self, boq_id: int) -> int:
        """Clear all embeddings for a specific BOQ"""
        query = """
            UPDATE store_boq_items
            SET description_embedding = NULL,
                embedding_generated_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE boq_id = %s AND description_embedding IS NOT NULL
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            affected = cur.rowcount
            conn.commit()
            return affected

    def _row_to_item(self, row) -> BOQItemWithEmbedding:
        """Convert database row to BOQItemWithEmbedding object"""
        return BOQItemWithEmbedding(
            item_id=row[0],
            boq_id=row[1],
            item_code=row[2],
            item_description=row[3],
            unit_of_measurement=row[4],
            quantity=row[5],
            supply_unit_rate=row[6],
            supply_amount=row[7],
            labour_unit_rate=row[8],
            labour_amount=row[9],
            total_amount=row[10],
            location_id=row[11],
            created_by=row[12],
            created_at=row[13],
            updated_by=row[14],
            updated_at=row[15],
            description_embedding=row[16] if len(row) > 16 else None,
            embedding_generated_at=row[17] if len(row) > 17 else None
        )