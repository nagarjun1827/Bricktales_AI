import psycopg2
from typing import List, Dict, Optional, Tuple
from core.settings import settings

class PriceRepository:
    """Repository for price fetching operations"""
    
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
    
    def get_boq_line_items(self, boq_id: int) -> List[Tuple]:
        """Get all line items for a BOQ"""
        query = """
            SELECT 
                item_id,
                item_code,
                item_description,
                unit_of_measurement,
                quantity
            FROM store_boq_items
            WHERE boq_id = %s
            ORDER BY item_id
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            return cur.fetchall()
    
    def find_similar_items(
        self,
        query_embedding: List[float],
        unit: str,
        limit: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """Find similar items from store_boq_items using vector similarity"""
        
        query_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        query = """
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
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                query, 
                (query_str, unit, query_str, min_similarity, query_str, limit)
            )
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    'item_id': row[0],
                    'item_code': row[1],
                    'description': row[2],
                    'unit': row[3],
                    'supply_rate': float(row[4]) if row[4] else 0.0,
                    'labour_rate': float(row[5]) if row[5] else 0.0,
                    'total_amount': float(row[6]) if row[6] else 0.0,
                    'project_name': row[7],
                    'file_name': row[8],
                    'similarity': float(row[9])
                })
            
            return results
    
    def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get BOQ file information"""
        query = """
            SELECT 
                bf.boq_id,
                bf.file_name,
                p.project_name,
                p.project_code
            FROM store_boq_files bf
            JOIN projects p ON bf.project_id = p.project_id
            WHERE bf.boq_id = %s
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    'boq_id': row[0],
                    'file_name': row[1],
                    'project_name': row[2],
                    'project_code': row[3]
                }
            return None