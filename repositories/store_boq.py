"""
Async Repository for store BOQ database operations.
Uses centralized async database connection.
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text, select, delete
from sqlalchemy import func
from datetime import datetime
from models.store_boq_models import StoreBoqProject, StoreBoqLocation, StoreBoqFile, StoreBoqItem
from models.project_models import Project
from connections.postgres_connection import DatabaseConnection

logger = logging.getLogger(__name__)

class StoreBOQRepository:
    """Async repository for store BOQ database operations."""
    
    def __init__(self):
        self.AsyncSessionLocal = DatabaseConnection.get_async_session_factory()

    async def insert_project(self, project_info) -> int:
        """Insert project asynchronously."""
        async with self.AsyncSessionLocal() as session:
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
            await session.commit()
            await session.refresh(project)
            return project.project_id

    async def insert_store_boq_project(self, store_project_info) -> int:
        """Insert store BOQ project asynchronously."""
        async with self.AsyncSessionLocal() as session:
            store_project = StoreBoqProject(
                project_id=store_project_info.project_id,
                store_project_name=store_project_info.store_project_name,
                store_project_code=store_project_info.store_project_code,
                total_project_value=store_project_info.total_project_value,
                created_by=store_project_info.created_by
            )
            session.add(store_project)
            await session.commit()
            await session.refresh(store_project)
            return store_project.store_project_id

    async def insert_location(self, location_info) -> int:
        """Insert location asynchronously."""
        async with self.AsyncSessionLocal() as session:
            location = StoreBoqLocation(
                store_project_id=location_info.store_project_id,
                location_name=location_info.location_name,
                address=location_info.address,
                latitude=location_info.latitude,
                longitude=location_info.longitude,
                created_by=location_info.created_by
            )
            session.add(location)
            await session.commit()
            await session.refresh(location)
            return location.location_id

    async def insert_boq_file(self, file_info) -> int:
        """Insert BOQ file asynchronously."""
        async with self.AsyncSessionLocal() as session:
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
            await session.commit()
            await session.refresh(boq_file)
            return boq_file.boq_id

    async def insert_boq_items_batch(self, items: List) -> None:
        """
        Batch insert BOQ items asynchronously.
        CRITICAL: Do NOT set supply_amount, labour_amount, or total_amount.
        These are GENERATED columns and PostgreSQL calculates them automatically.
        """
        if not items:
            return
        
        logger.info(f"Inserting {len(items)} items (computed columns will be auto-calculated)")
        
        async with self.AsyncSessionLocal() as session:
            for item in items:
                boq_item = StoreBoqItem(
                    boq_id=item.boq_id,
                    item_code=item.item_code,
                    item_description=item.item_description,
                    unit_of_measurement=item.unit_of_measurement,
                    quantity=item.quantity,
                    supply_unit_rate=item.supply_unit_rate if item.supply_unit_rate is not None else 0.0,
                    labour_unit_rate=item.labour_unit_rate if item.labour_unit_rate is not None else 0.0,
                    location_id=item.location_id,
                    created_by=item.created_by
                )
                session.add(boq_item)
            
            await session.commit()
            logger.info(f"Successfully inserted {len(items)} items")

    async def get_boq_totals(self, boq_id: int, items: List = None) -> Dict[str, float]:
        """Get BOQ totals asynchronously."""
        if items:
            # Calculate from items in memory
            return {
                "item_count": len(items),
                "total_supply": sum(item.supply_amount for item in items),
                "total_labour": sum(item.labour_amount for item in items),
                "total_amount": sum(item.total_amount for item in items),
            }
        
        # Query from database using SQLAlchemy ORM
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    func.count(StoreBoqItem.item_id).label('item_count'),
                    func.coalesce(func.sum(StoreBoqItem.supply_amount), 0).label('total_supply'),
                    func.coalesce(func.sum(StoreBoqItem.labour_amount), 0).label('total_labour'),
                    func.coalesce(func.sum(StoreBoqItem.total_amount), 0).label('total_amount')
                ).where(StoreBoqItem.boq_id == boq_id)
            )
            row = result.fetchone()
            
            return {
                "item_count": row.item_count,
                "total_supply": float(row.total_supply),
                "total_labour": float(row.total_labour),
                "total_amount": float(row.total_amount),
            }
    
    async def ensure_vector_extension(self):
        """Ensure pgvector extension asynchronously."""
        async with self.AsyncSessionLocal() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await session.commit()
    
    async def get_items_without_embeddings(self, boq_id: int) -> List:
        """Get items without embeddings asynchronously."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(StoreBoqItem).where(
                    StoreBoqItem.boq_id == boq_id,
                    StoreBoqItem.description_embedding.is_(None)
                )
            )
            items = result.scalars().all()
            return list(items)
    
    async def update_embeddings_batch(self, item_ids: List[int], embeddings: List[List[float]]) -> int:
        """
        Update embeddings for a batch of items asynchronously using SQLAlchemy ORM.
        This avoids SQL syntax errors with vector type casting and parameter binding.
        """
        if not item_ids or not embeddings:
            return 0
        
        updated = 0
        async with self.AsyncSessionLocal() as session:
            for item_id, embedding in zip(item_ids, embeddings):
                try:
                    # Get the item using async ORM
                    result = await session.execute(
                        select(StoreBoqItem).where(StoreBoqItem.item_id == item_id)
                    )
                    item = result.scalar_one_or_none()
                    
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
            await session.commit()
        
        return updated
    
    async def delete_boq_by_id(self, boq_id: int) -> Dict[str, Any]:
        """Delete BOQ and related data asynchronously."""
        try:
            async with self.AsyncSessionLocal() as session:
                deleted_counts = {}
                
                # Get BOQ info
                result = await session.execute(
                    select(StoreBoqFile).where(StoreBoqFile.boq_id == boq_id)
                )
                boq = result.scalar_one_or_none()
                
                if not boq:
                    return {'success': False, 'error': f'BOQ {boq_id} not found'}
                
                store_project_id = boq.store_project_id
                
                # Delete items
                result = await session.execute(
                    delete(StoreBoqItem).where(StoreBoqItem.boq_id == boq_id)
                )
                deleted_counts['boq_items'] = result.rowcount
                
                # Delete BOQ file
                await session.delete(boq)
                deleted_counts['boq_file'] = 1
                
                await session.commit()
                
                return {
                    'success': True,
                    'boq_id': boq_id,
                    'deleted_counts': deleted_counts,
                    'message': f'Stored BOQ deleted successfully'
                }
                
        except Exception as e:
            logger.error(f"Delete error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    async def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get BOQ info asynchronously."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    StoreBoqFile.boq_id,
                    StoreBoqFile.file_name,
                    StoreBoqFile.store_project_id,
                    Project.project_name,
                    Project.project_code,
                    StoreBoqFile.created_at
                )
                .join(StoreBoqProject, StoreBoqFile.store_project_id == StoreBoqProject.store_project_id)
                .join(Project, StoreBoqProject.project_id == Project.project_id)
                .where(StoreBoqFile.boq_id == boq_id)
            )
            row = result.fetchone()
            
            if row:
                return {
                    'boq_id': row.boq_id,
                    'file_name': row.file_name,
                    'store_project_id': row.store_project_id,
                    'project_name': row.project_name,
                    'project_code': row.project_code,
                    'created_at': row.created_at.isoformat() if row.created_at else None
                }
            return None