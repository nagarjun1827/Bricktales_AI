"""
Repository for price fetching operations using SQLAlchemy.
"""
import logging
from typing import List, Dict, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session

from connections.postgres_connection import get_db_session
from models.store_boq_models import StoreBoqItem, StoreBoqFile, StoreBoqProject
from models.project_models import Project

logger = logging.getLogger(__name__)


class PriceRepository:
    """Repository for price fetching operations using SQLAlchemy."""
    
    def __init__(self):
        """Initialize repository."""
        pass
    
    def _get_session(self) -> Session:
        """Get a database session."""
        return get_db_session()
    
    def get_boq_line_items(self, boq_id: int) -> List[Tuple]:
        """Get all line items for a BOQ using SQLAlchemy."""
        session = self._get_session()
        try:
            items = session.query(
                StoreBoqItem.item_id,
                StoreBoqItem.item_code,
                StoreBoqItem.item_description,
                StoreBoqItem.unit_of_measurement,
                StoreBoqItem.quantity
            ).filter(
                StoreBoqItem.boq_id == boq_id
            ).order_by(
                StoreBoqItem.item_id
            ).all()
            
            return items
        finally:
            session.close()
    
    def find_similar_items(
        self,
        query_embedding: List[float],
        unit: str,
        limit: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Find similar items from store_boq_items using vector similarity.
        
        Note: Uses raw SQL for pgvector operations as SQLAlchemy doesn't have
        native support for vector similarity operators.
        """
        session = self._get_session()
        try:
            query_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Use raw SQL for vector similarity search
            result = session.execute(
                text("""
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
                        1 - (bi.description_embedding <=> :query_embedding::vector) as similarity
                    FROM store_boq_items bi
                    JOIN store_boq_files bf ON bi.boq_id = bf.boq_id
                    JOIN store_boq_projects sbp ON bf.store_project_id = sbp.store_project_id
                    JOIN projects p ON sbp.project_id = p.project_id
                    WHERE bi.description_embedding IS NOT NULL
                      AND bi.unit_of_measurement = :unit
                      AND (1 - (bi.description_embedding <=> :query_embedding::vector)) >= :min_similarity
                      AND bi.supply_unit_rate IS NOT NULL
                      AND bi.supply_unit_rate > 0
                    ORDER BY bi.description_embedding <=> :query_embedding::vector
                    LIMIT :limit
                """),
                {
                    'query_embedding': query_str,
                    'unit': unit,
                    'min_similarity': min_similarity,
                    'limit': limit
                }
            )
            
            rows = result.fetchall()
            
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
            
        finally:
            session.close()
    
    def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get BOQ file information using SQLAlchemy."""
        session = self._get_session()
        try:
            result = session.query(
                StoreBoqFile.boq_id,
                StoreBoqFile.file_name,
                Project.project_name,
                Project.project_code
            ).join(
                StoreBoqProject, 
                StoreBoqFile.store_project_id == StoreBoqProject.store_project_id
            ).join(
                Project, StoreBoqProject.project_id == Project.project_id
            ).filter(
                StoreBoqFile.boq_id == boq_id
            ).first()
            
            if result:
                return {
                    'boq_id': result.boq_id,
                    'file_name': result.file_name,
                    'project_name': result.project_name,
                    'project_code': result.project_code
                }
            return None
        finally:
            session.close()