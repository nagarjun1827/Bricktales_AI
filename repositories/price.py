"""
Repository for price fetching operations using SQLAlchemy ORM.
Uses SQLAlchemy queries instead of raw SQL.
"""
import logging
from typing import List, Dict, Optional, Tuple
from sqlalchemy import text, func, literal_column
from sqlalchemy.orm import Session

from connections.postgres_connection import get_db_session
from models.store_boq_models import StoreBoqItem, StoreBoqFile, StoreBoqProject
from models.project_models import Project

logger = logging.getLogger(__name__)


class PriceRepository:
    """Repository for price fetching operations using SQLAlchemy ORM."""
    
    def __init__(self):
        """Initialize repository."""
        pass
    
    def _get_session(self) -> Session:
        """Get a database session."""
        return get_db_session()
    
    def get_boq_line_items(self, boq_id: int) -> List[Tuple]:
        """Get all line items for a BOQ using SQLAlchemy ORM."""
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
        
        Uses SQLAlchemy ORM with pgvector operator for similarity search.
        """
        session = self._get_session()
        try:
            query_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Use SQLAlchemy with text() for vector operations
            # Note: pgvector operators (<=> for cosine distance) need to be in text()
            similarity = literal_column(
                f"1 - (description_embedding <=> '{query_str}'::vector)"
            ).label('similarity')
            
            # Build query using SQLAlchemy ORM
            query = session.query(
                StoreBoqItem.item_id,
                StoreBoqItem.item_code,
                StoreBoqItem.item_description,
                StoreBoqItem.unit_of_measurement,
                StoreBoqItem.supply_unit_rate,
                StoreBoqItem.labour_unit_rate,
                StoreBoqItem.supply_amount,      # Can now read computed column
                StoreBoqItem.labour_amount,      # Can now read computed column
                StoreBoqItem.total_amount,       # Can now read computed column
                Project.project_name,
                StoreBoqFile.file_name,
                similarity
            ).join(
                StoreBoqFile, StoreBoqItem.boq_id == StoreBoqFile.boq_id
            ).join(
                StoreBoqProject, StoreBoqFile.store_project_id == StoreBoqProject.store_project_id
            ).join(
                Project, StoreBoqProject.project_id == Project.project_id
            ).filter(
                StoreBoqItem.description_embedding.isnot(None),
                StoreBoqItem.unit_of_measurement == unit,
                literal_column(f"1 - (description_embedding <=> '{query_str}'::vector)") >= min_similarity,
                StoreBoqItem.supply_unit_rate.isnot(None),
                StoreBoqItem.supply_unit_rate > 0
            ).order_by(
                literal_column(f"description_embedding <=> '{query_str}'::vector")
            ).limit(limit)
            
            rows = query.all()
            
            results = []
            for row in rows:
                results.append({
                    'item_id': row[0],
                    'item_code': row[1],
                    'description': row[2],
                    'unit': row[3],
                    'supply_rate': float(row[4]) if row[4] else 0.0,
                    'labour_rate': float(row[5]) if row[5] else 0.0,
                    'supply_amount': float(row[6]) if row[6] else 0.0,
                    'labour_amount': float(row[7]) if row[7] else 0.0,
                    'total_amount': float(row[8]) if row[8] else 0.0,
                    'project_name': row[9],
                    'file_name': row[10],
                    'similarity': float(row[11])
                })
            
            return results
            
        finally:
            session.close()
    
    def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get BOQ file information using SQLAlchemy ORM."""
        session = self._get_session()
        try:
            result = session.query(
                StoreBoqFile.boq_id,
                StoreBoqFile.file_name,
                Project.project_name,
                Project.project_code
            ).join(
                StoreBoqProject, StoreBoqFile.store_project_id == StoreBoqProject.store_project_id
            ).join(
                Project, StoreBoqProject.project_id == Project.project_id
            ).filter(
                StoreBoqFile.boq_id == boq_id
            ).first()
            
            if result:
                return {
                    'boq_id': result[0],
                    'file_name': result[1],
                    'project_name': result[2],
                    'project_code': result[3]
                }
            return None
        finally:
            session.close()