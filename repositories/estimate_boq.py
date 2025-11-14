"""
Async Repository for To-Be-Estimated BOQ operations using SQLAlchemy.
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import func, select, delete
from connections.postgres_connection import DatabaseConnection
from models.project_models import Project
from models.estimate_boq_models import EstimateBoqProject, EstimateBoqLocation, EstimateBoqFile, EstimateBoqItem
from models.store_boq_models import StoreBoqProject

logger = logging.getLogger(__name__)

class TBEProjectInfo:
    def __init__(self, project_name, project_code, project_type, client_name=None,
                 start_date=None, end_date=None, version=1, created_by="system"):
        self.project_name = project_name
        self.project_code = project_code
        self.project_type = project_type
        self.client_name = client_name
        self.start_date = start_date
        self.end_date = end_date
        self.version = version
        self.created_by = created_by

class EstimateBOQProjectInfo:
    def __init__(self, project_id, estimate_project_name, estimate_project_code,
                 estimation_status="pending", estimated_value=None, estimated_by=None,
                 estimated_at=None, created_by="system"):
        self.project_id = project_id
        self.estimate_project_name = estimate_project_name
        self.estimate_project_code = estimate_project_code
        self.estimation_status = estimation_status
        self.estimated_value = estimated_value
        self.estimated_by = estimated_by
        self.estimated_at = estimated_at
        self.created_by = created_by

class TBELocationInfo:
    def __init__(self, estimate_project_id, location_name, address=None,
                 latitude=None, longitude=None, created_by="system"):
        self.estimate_project_id = estimate_project_id
        self.location_name = location_name
        self.address = address
        self.latitude = latitude
        self.longitude = longitude
        self.created_by = created_by


class TBEBOQFileInfo:
    def __init__(self, estimate_project_id, file_name, file_path,
                 file_type="xlsx", version=1, is_active=True, created_by="system"):
        self.estimate_project_id = estimate_project_id
        self.file_name = file_name
        self.file_path = file_path
        self.file_type = file_type
        self.version = version
        self.is_active = is_active
        self.created_by = created_by

class TBEBOQItem:
    def __init__(self, boq_id, item_description, unit_of_measurement, quantity,
                 location_id, item_code=None, created_by="system"):
        self.boq_id = boq_id
        self.item_code = item_code
        self.item_description = item_description
        self.unit_of_measurement = unit_of_measurement
        self.quantity = quantity
        self.location_id = location_id
        self.created_by = created_by


class TBEBOQRepository:
    """Async repository for To-Be-Estimated BOQ operations using SQLAlchemy."""
    
    def __init__(self):
        """Initialize repository."""
        self.AsyncSessionLocal = DatabaseConnection.get_async_session_factory()

    async def insert_project(self, project_info: TBEProjectInfo) -> int:
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
            project_id = project.project_id
            logger.info(f"Inserted project with ID: {project_id}")
            return project_id

    async def insert_estimate_boq_project(self, estimate_project_info: EstimateBOQProjectInfo) -> int:
        """Insert estimate BOQ project asynchronously."""
        async with self.AsyncSessionLocal() as session:
            estimate_project = EstimateBoqProject(
                project_id=estimate_project_info.project_id,
                estimate_project_name=estimate_project_info.estimate_project_name,
                estimate_project_code=estimate_project_info.estimate_project_code,
                estimation_status=estimate_project_info.estimation_status,
                estimated_value=estimate_project_info.estimated_value,
                estimated_by=estimate_project_info.estimated_by,
                estimated_at=estimate_project_info.estimated_at,
                created_by=estimate_project_info.created_by
            )
            session.add(estimate_project)
            await session.commit()
            await session.refresh(estimate_project)
            estimate_project_id = estimate_project.estimate_project_id
            logger.info(f"Inserted estimate project with ID: {estimate_project_id}")
            return estimate_project_id

    async def insert_location(self, location_info: TBELocationInfo) -> int:
        """Insert location asynchronously."""
        async with self.AsyncSessionLocal() as session:
            location = EstimateBoqLocation(
                estimate_project_id=location_info.estimate_project_id,
                location_name=location_info.location_name,
                address=location_info.address,
                latitude=location_info.latitude,
                longitude=location_info.longitude,
                created_by=location_info.created_by
            )
            session.add(location)
            await session.commit()
            await session.refresh(location)
            location_id = location.location_id
            logger.info(f"Inserted location with ID: {location_id}")
            return location_id

    async def insert_tbe_boq_file(self, file_info: TBEBOQFileInfo) -> int:
        """Insert to-be-estimated BOQ file record asynchronously."""
        async with self.AsyncSessionLocal() as session:
            boq_file = EstimateBoqFile(
                estimate_project_id=file_info.estimate_project_id,
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
            boq_id = boq_file.boq_id
            logger.info(f"Inserted TBE BOQ file with ID: {boq_id}")
            return boq_id

    async def insert_tbe_items_batch(self, items: List[TBEBOQItem]) -> None:
        """Batch insert to-be-estimated BOQ items asynchronously."""
        if not items:
            logger.warning("No items to insert")
            return
        
        logger.info(f"Preparing to insert {len(items)} items")
        
        async with self.AsyncSessionLocal() as session:
            tbe_items = []
            for item in items:
                tbe_item = EstimateBoqItem(
                    boq_id=item.boq_id,
                    item_code=item.item_code,
                    item_description=item.item_description,
                    unit_of_measurement=item.unit_of_measurement,
                    quantity=item.quantity,
                    location_id=item.location_id,
                    created_by=item.created_by
                )
                tbe_items.append(tbe_item)
            
            # Add all items
            session.add_all(tbe_items)
            await session.commit()
            logger.info(f"Successfully inserted {len(items)} items")

    async def get_tbe_boq_summary(self, boq_id: int) -> Dict[str, int]:
        """Get summary of to-be-estimated BOQ asynchronously."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(func.count(EstimateBoqItem.item_id))
                .where(EstimateBoqItem.boq_id == boq_id)
            )
            item_count = result.scalar()
            
            return {
                "item_count": item_count if item_count else 0
            }

    async def get_tbe_items_by_boq(self, boq_id: int, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get TBE items for a BOQ with pagination asynchronously."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(EstimateBoqItem)
                .where(EstimateBoqItem.boq_id == boq_id)
                .order_by(EstimateBoqItem.item_id)
                .limit(limit)
                .offset(offset)
            )
            items = result.scalars().all()
            
            result_list = []
            for item in items:
                result_list.append({
                    "item_id": item.item_id,
                    "boq_id": item.boq_id,
                    "item_code": item.item_code,
                    "item_description": item.item_description,
                    "unit_of_measurement": item.unit_of_measurement,
                    "quantity": float(item.quantity),
                    "location_id": item.location_id,
                    "created_by": item.created_by,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                    "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                })
            
            return result_list
        
    async def delete_boq_by_id(self, boq_id: int) -> Dict[str, Any]:
        """Delete Estimate BOQ file and all related data asynchronously."""
        async with self.AsyncSessionLocal() as session:
            try:
                deleted_counts = {}
                
                # Get BOQ info before deletion
                result = await session.execute(
                    select(EstimateBoqFile).where(EstimateBoqFile.boq_id == boq_id)
                )
                boq_file = result.scalar_one_or_none()
                
                if not boq_file:
                    return {
                        'success': False,
                        'error': f'Estimate BOQ file with ID {boq_id} not found'
                    }
                
                estimate_project_id = boq_file.estimate_project_id
                file_name = boq_file.file_name
                
                logger.info(f"Deleting Estimate BOQ: {file_name} (ID: {boq_id})")
                
                # Delete BOQ items
                result = await session.execute(
                    delete(EstimateBoqItem).where(EstimateBoqItem.boq_id == boq_id)
                )
                deleted_counts['boq_items'] = result.rowcount
                logger.info(f"Deleted {result.rowcount} BOQ items")
                
                # Delete BOQ file record
                await session.delete(boq_file)
                deleted_counts['boq_file'] = 1
                logger.info("Deleted BOQ file record")
                
                # Check if estimate project has other BOQ files
                result = await session.execute(
                    select(func.count(EstimateBoqFile.boq_id))
                    .where(EstimateBoqFile.estimate_project_id == estimate_project_id)
                )
                remaining_boqs = result.scalar()
                
                if remaining_boqs == 0:
                    logger.info(f"No other BOQs found for estimate project {estimate_project_id}")
                    
                    # Get estimate project
                    result = await session.execute(
                        select(EstimateBoqProject)
                        .where(EstimateBoqProject.estimate_project_id == estimate_project_id)
                    )
                    estimate_project = result.scalar_one_or_none()
                    
                    if estimate_project:
                        project_id = estimate_project.project_id
                        
                        # Delete locations
                        result = await session.execute(
                            delete(EstimateBoqLocation)
                            .where(EstimateBoqLocation.estimate_project_id == estimate_project_id)
                        )
                        deleted_counts['locations'] = result.rowcount
                        logger.info(f"Deleted {result.rowcount} locations")
                        
                        # Delete estimate project
                        await session.delete(estimate_project)
                        deleted_counts['estimate_project'] = 1
                        logger.info("Deleted estimate project")
                        
                        # Check if project has other store/estimate projects
                        result_store = await session.execute(
                            select(func.count(StoreBoqProject.store_project_id))
                            .where(StoreBoqProject.project_id == project_id)
                        )
                        store_count = result_store.scalar()
                        
                        result_estimate = await session.execute(
                            select(func.count(EstimateBoqProject.estimate_project_id))
                            .where(EstimateBoqProject.project_id == project_id)
                        )
                        estimate_count = result_estimate.scalar()
                        
                        remaining_projects = store_count + estimate_count
                        
                        if remaining_projects == 0:
                            # Delete main project
                            result = await session.execute(
                                select(Project).where(Project.project_id == project_id)
                            )
                            project = result.scalar_one_or_none()
                            if project:
                                await session.delete(project)
                                deleted_counts['project'] = 1
                                logger.info("Deleted main project")
                        else:
                            logger.info("Main project retained (has other references)")
                            deleted_counts['project'] = 0
                    else:
                        deleted_counts['locations'] = 0
                        deleted_counts['estimate_project'] = 0
                        deleted_counts['project'] = 0
                else:
                    logger.info(f"Estimate project retained ({remaining_boqs} other BOQ(s) exist)")
                    deleted_counts['locations'] = 0
                    deleted_counts['estimate_project'] = 0
                    deleted_counts['project'] = 0
                
                await session.commit()
                logger.info(f"Successfully deleted Estimate BOQ {boq_id}")
                
                return {
                    'success': True,
                    'boq_id': boq_id,
                    'deleted_counts': deleted_counts,
                    'message': f'Estimate BOQ deleted successfully'
                }
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error deleting Estimate BOQ: {e}", exc_info=True)
                return {
                    'success': False,
                    'error': str(e)
                }

    async def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get Estimate BOQ file information asynchronously."""
        async with self.AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    EstimateBoqFile.boq_id,
                    EstimateBoqFile.file_name,
                    EstimateBoqFile.estimate_project_id,
                    Project.project_name,
                    Project.project_code,
                    EstimateBoqFile.created_at
                )
                .join(EstimateBoqProject, EstimateBoqFile.estimate_project_id == EstimateBoqProject.estimate_project_id)
                .join(Project, EstimateBoqProject.project_id == Project.project_id)
                .where(EstimateBoqFile.boq_id == boq_id)
            )
            row = result.fetchone()
            
            if row:
                return {
                    'boq_id': row.boq_id,
                    'file_name': row.file_name,
                    'estimate_project_id': row.estimate_project_id,
                    'project_name': row.project_name,
                    'project_code': row.project_code,
                    'created_at': row.created_at.isoformat() if row.created_at else None
                }
            return None