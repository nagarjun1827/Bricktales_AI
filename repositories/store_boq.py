"""
Repository for store BOQ database operations.
FIXED: Does not insert computed columns (supply_amount, labour_amount, total_amount)
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, Session
from models.store_boq_models import StoreBoqProject, StoreBoqLocation, StoreBoqFile, StoreBoqItem
from models.project_models import Project
from core.settings import settings

logger = logging.getLogger(__name__)


class StoreBOQRepository:
    """Repository for store BOQ database operations."""
    
    def __init__(self):
        # SQLAlchemy setup
        from sqlalchemy import create_engine
        db_url = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _get_session(self) -> Session:
        """Get SQLAlchemy session."""
        return self.SessionLocal()

    def insert_project(self, project_info) -> int:
        """Insert project."""
        with self._get_session() as session:
            project = Project(
                project_name=project_info.project_name,
                project_code=project_info.project_code,
                project_type=project_info.project_type,
                client_name=project_info.client_name,
                start_date=project_info.start_date,
                end_date=project_info.end_date,
                version=project_info.version,
                created_by=project_info.created_by
            )
            session.add(project)
            session.commit()
            session.refresh(project)
            return project.project_id

    def insert_store_boq_project(self, store_project_info) -> int:
        """Insert store BOQ project."""
        with self._get_session() as session:
            store_project = StoreBoqProject(
                project_id=store_project_info.project_id,
                store_project_name=store_project_info.store_project_name,
                store_project_code=store_project_info.store_project_code,
                total_project_value=store_project_info.total_project_value,
                created_by=store_project_info.created_by
            )
            session.add(store_project)
            session.commit()
            session.refresh(store_project)
            return store_project.store_project_id

    def insert_location(self, location_info) -> int:
        """Insert location."""
        with self._get_session() as session:
            location = StoreBoqLocation(
                store_project_id=location_info.store_project_id,
                location_name=location_info.location_name,
                address=location_info.address,
                latitude=location_info.latitude,
                longitude=location_info.longitude,
                created_by=location_info.created_by
            )
            session.add(location)
            session.commit()
            session.refresh(location)
            return location.location_id

    def insert_boq_file(self, file_info) -> int:
        """Insert BOQ file."""
        with self._get_session() as session:
            boq_file = StoreBoqFile(
                store_project_id=file_info.store_project_id,
                file_name=file_info.file_name,
                file_path=file_info.file_path,
                file_type=file_info.file_type,
                version=file_info.version,
                is_active=file_info.is_active,
                created_by=file_info.created_by
            )
            session.add(boq_file)
            session.commit()
            session.refresh(boq_file)
            return boq_file.boq_id

    def insert_boq_items_batch(self, items: List) -> None:
        """
        Batch insert BOQ items.
        
        CRITICAL: Do NOT set supply_amount, labour_amount, or total_amount.
        These are GENERATED columns and PostgreSQL calculates them automatically.
        """
        if not items:
            return
        
        logger.info(f"Inserting {len(items)} items (computed columns will be auto-calculated)")
        
        with self._get_session() as session:
            for item in items:
                # IMPORTANT: Only set the base columns, not the computed ones
                boq_item = StoreBoqItem(
                    boq_id=item.boq_id,
                    item_code=item.item_code,
                    item_description=item.item_description,
                    unit_of_measurement=item.unit_of_measurement,
                    quantity=item.quantity,
                    # Ensure rates are never NULL (use 0.0 as default)
                    supply_unit_rate=item.supply_unit_rate if item.supply_unit_rate is not None else 0.0,
                    labour_unit_rate=item.labour_unit_rate if item.labour_unit_rate is not None else 0.0,
                    location_id=item.location_id,
                    created_by=item.created_by
                    # ✅ DO NOT SET: supply_amount, labour_amount, total_amount
                    # PostgreSQL will calculate these automatically
                )
                session.add(boq_item)
            
            session.commit()
            logger.info(f"Successfully inserted {len(items)} items")

    def get_boq_totals(self, boq_id: int, items: List = None) -> Dict[str, float]:
        """Get BOQ totals."""
        if items:
            # Calculate from items in memory
            return {
                "item_count": len(items),
                "total_supply": sum(item.supply_amount for item in items),
                "total_labour": sum(item.labour_amount for item in items),
                "total_amount": sum(item.total_amount for item in items),
            }
        
        # Query from database (amounts are now computed)
        with self._get_session() as session:
            result = session.execute(
                text("""
                    SELECT 
                        COUNT(*) as item_count,
                        COALESCE(SUM(supply_amount), 0) as total_supply,
                        COALESCE(SUM(labour_amount), 0) as total_labour,
                        COALESCE(SUM(total_amount), 0) as total_amount
                    FROM store_boq_items
                    WHERE boq_id = :boq_id
                """),
                {"boq_id": boq_id}
            ).fetchone()
            
            return {
                "item_count": result[0],
                "total_supply": float(result[1]),
                "total_labour": float(result[2]),
                "total_amount": float(result[3]),
            }
    
    def ensure_vector_extension(self):
        """Ensure pgvector extension."""
        with self._get_session() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
    
    def get_items_without_embeddings(self, boq_id: int) -> List:
        """Get items without embeddings."""
        with self._get_session() as session:
            items = session.query(StoreBoqItem).filter(
                StoreBoqItem.boq_id == boq_id,
                StoreBoqItem.description_embedding.is_(None)
            ).all()
            return items
    
    def update_embeddings_batch(self, item_ids: List[int], embeddings: List[List[float]]) -> int:
        """
        Update embeddings for a batch of items using SQLAlchemy ORM.
        
        This avoids SQL syntax errors with vector type casting and parameter binding.
        """
        if not item_ids or not embeddings:
            return 0
        
        updated = 0
        with self._get_session() as session:
            for item_id, embedding in zip(item_ids, embeddings):
                try:
                    from datetime import datetime
                    
                    # Get the item using ORM
                    item = session.query(StoreBoqItem).filter(
                        StoreBoqItem.item_id == item_id
                    ).first()
                    
                    if item:
                        # Update using ORM - SQLAlchemy handles vector type conversion
                        item.description_embedding = embedding
                        item.embedding_generated_at = datetime.now()
                        item.updated_at = datetime.now()
                        updated += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to update item {item_id}: {e}")
                    continue
            
            # Commit all updates at once
            session.commit()
        
        return updated
    
    def delete_boq_by_id(self, boq_id: int) -> Dict[str, Any]:
        """Delete BOQ and related data."""
        try:
            with self._get_session() as session:
                deleted_counts = {}
                
                # Get BOQ info
                boq = session.query(StoreBoqFile).filter(
                    StoreBoqFile.boq_id == boq_id
                ).first()
                
                if not boq:
                    return {'success': False, 'error': f'BOQ {boq_id} not found'}
                
                store_project_id = boq.store_project_id
                
                # Delete items
                items_deleted = session.query(StoreBoqItem).filter(
                    StoreBoqItem.boq_id == boq_id
                ).delete()
                deleted_counts['boq_items'] = items_deleted
                
                # Delete BOQ file
                session.delete(boq)
                deleted_counts['boq_file'] = 1
                
                session.commit()
                
                return {
                    'success': True,
                    'boq_id': boq_id,
                    'deleted_counts': deleted_counts,
                    'message': f'BOQ {boq_id} deleted successfully'
                }
                
        except Exception as e:
            logger.error(f"Delete error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get BOQ info."""
        with self._get_session() as session:
            result = session.execute(
                text("""
                    SELECT 
                        sbf.boq_id, sbf.file_name, sbf.store_project_id,
                        p.project_name, p.project_code, sbf.created_at
                    FROM store_boq_files sbf
                    JOIN store_boq_projects sbp ON sbf.store_project_id = sbp.store_project_id
                    JOIN projects p ON sbp.project_id = p.project_id
                    WHERE sbf.boq_id = :boq_id
                """),
                {"boq_id": boq_id}
            ).fetchone()
            
            if result:
                return {
                    'boq_id': result[0],
                    'file_name': result[1],
                    'store_project_id': result[2],
                    'project_name': result[3],
                    'project_code': result[4],
                    'created_at': result[5].isoformat() if result[5] else None
                }
            return None