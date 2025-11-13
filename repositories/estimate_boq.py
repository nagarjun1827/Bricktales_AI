"""
Repository for To-Be-Estimated BOQ operations using SQLAlchemy.
"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import func
from sqlalchemy.orm import Session
from connections.postgres_connection import get_db_session
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
    """Repository for To-Be-Estimated BOQ operations using SQLAlchemy."""
    
    def __init__(self):
        """Initialize repository."""
        pass
    
    def _get_session(self) -> Session:
        """Get a database session."""
        return get_db_session()

    def insert_project(self, project_info: TBEProjectInfo) -> int:
        """Insert project using SQLAlchemy."""
        session = self._get_session()
        try:
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
            project_id = project.project_id
            logger.info(f"Inserted project with ID: {project_id}")
            return project_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert project: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def insert_estimate_boq_project(self, estimate_project_info: EstimateBOQProjectInfo) -> int:
        """Insert estimate BOQ project using SQLAlchemy."""
        session = self._get_session()
        try:
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
            session.commit()
            session.refresh(estimate_project)
            estimate_project_id = estimate_project.estimate_project_id
            logger.info(f"Inserted estimate project with ID: {estimate_project_id}")
            return estimate_project_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert estimate project: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def insert_location(self, location_info: TBELocationInfo) -> int:
        """Insert location using SQLAlchemy."""
        session = self._get_session()
        try:
            location = EstimateBoqLocation(
                estimate_project_id=location_info.estimate_project_id,
                location_name=location_info.location_name,
                address=location_info.address,
                latitude=location_info.latitude,
                longitude=location_info.longitude,
                created_by=location_info.created_by
            )
            session.add(location)
            session.commit()
            session.refresh(location)
            location_id = location.location_id
            logger.info(f"Inserted location with ID: {location_id}")
            return location_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert location: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def insert_tbe_boq_file(self, file_info: TBEBOQFileInfo) -> int:
        """Insert to-be-estimated BOQ file record using SQLAlchemy."""
        session = self._get_session()
        try:
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
            session.commit()
            session.refresh(boq_file)
            boq_id = boq_file.boq_id
            logger.info(f"Inserted TBE BOQ file with ID: {boq_id}")
            return boq_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert TBE BOQ file: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def insert_tbe_items_batch(self, items: List[TBEBOQItem]) -> None:
        """Batch insert to-be-estimated BOQ items using SQLAlchemy."""
        if not items:
            logger.warning("No items to insert")
            return
        
        logger.info(f"Preparing to insert {len(items)} items")
        
        session = self._get_session()
        try:
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
            
            # Bulk insert
            session.bulk_save_objects(tbe_items)
            session.commit()
            logger.info(f"Successfully inserted {len(items)} items")
        except Exception as e:
            session.rollback()
            logger.error(f"Insert failed: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def get_tbe_boq_summary(self, boq_id: int) -> Dict[str, int]:
        """Get summary of to-be-estimated BOQ using SQLAlchemy."""
        session = self._get_session()
        try:
            item_count = session.query(func.count(EstimateBoqItem.item_id)).filter(
                EstimateBoqItem.boq_id == boq_id
            ).scalar()
            
            return {
                "item_count": item_count if item_count else 0
            }
        finally:
            session.close()

    def get_tbe_items_by_boq(self, boq_id: int, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get TBE items for a BOQ with pagination using SQLAlchemy."""
        session = self._get_session()
        try:
            items = session.query(EstimateBoqItem).filter(
                EstimateBoqItem.boq_id == boq_id
            ).order_by(
                EstimateBoqItem.item_id
            ).limit(limit).offset(offset).all()
            
            result = []
            for item in items:
                result.append({
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
            
            return result
        finally:
            session.close()
        
    def delete_boq_by_id(self, boq_id: int) -> Dict[str, Any]:
        """Delete Estimate BOQ file and all related data using SQLAlchemy."""
        session = self._get_session()
        try:
            deleted_counts = {}
            
            # Get BOQ info before deletion
            boq_file = session.query(EstimateBoqFile).filter(
                EstimateBoqFile.boq_id == boq_id
            ).first()
            
            if not boq_file:
                return {
                    'success': False,
                    'error': f'Estimate BOQ file with ID {boq_id} not found'
                }
            
            estimate_project_id = boq_file.estimate_project_id
            file_name = boq_file.file_name
            
            logger.info(f"Deleting Estimate BOQ: {file_name} (ID: {boq_id})")
            
            # Delete BOQ items
            items_deleted = session.query(EstimateBoqItem).filter(
                EstimateBoqItem.boq_id == boq_id
            ).delete()
            deleted_counts['boq_items'] = items_deleted
            logger.info(f"Deleted {items_deleted} BOQ items")
            
            # Delete BOQ file record
            session.delete(boq_file)
            deleted_counts['boq_file'] = 1
            logger.info("Deleted BOQ file record")
            
            # Check if estimate project has other BOQ files
            remaining_boqs = session.query(func.count(EstimateBoqFile.boq_id)).filter(
                EstimateBoqFile.estimate_project_id == estimate_project_id
            ).scalar()
            
            if remaining_boqs == 0:
                logger.info(f"No other BOQs found for estimate project {estimate_project_id}")
                
                # Get estimate project
                estimate_project = session.query(EstimateBoqProject).filter(
                    EstimateBoqProject.estimate_project_id == estimate_project_id
                ).first()
                
                if estimate_project:
                    project_id = estimate_project.project_id
                    
                    # Delete locations
                    locations_deleted = session.query(EstimateBoqLocation).filter(
                        EstimateBoqLocation.estimate_project_id == estimate_project_id
                    ).delete()
                    deleted_counts['locations'] = locations_deleted
                    logger.info(f"Deleted {locations_deleted} locations")
                    
                    # Delete estimate project
                    session.delete(estimate_project)
                    deleted_counts['estimate_project'] = 1
                    logger.info("Deleted estimate project")
                    
                    # Check if project has other store/estimate projects
                    remaining_projects = (
                        session.query(func.count(StoreBoqProject.store_project_id))
                        .filter(StoreBoqProject.project_id == project_id).scalar() +
                        session.query(func.count(EstimateBoqProject.estimate_project_id))
                        .filter(EstimateBoqProject.project_id == project_id).scalar()
                    )
                    
                    if remaining_projects == 0:
                        # Delete main project
                        project = session.query(Project).filter(
                            Project.project_id == project_id
                        ).first()
                        if project:
                            session.delete(project)
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
            
            session.commit()
            logger.info(f"Successfully deleted Estimate BOQ {boq_id}")
            
            return {
                'success': True,
                'boq_id': boq_id,
                'deleted_counts': deleted_counts,
                'message': f'Estimate BOQ {boq_id} and related data deleted successfully'
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting Estimate BOQ: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            session.close()

    def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get Estimate BOQ file information using SQLAlchemy."""
        session = self._get_session()
        try:
            result = session.query(
                EstimateBoqFile.boq_id,
                EstimateBoqFile.file_name,
                EstimateBoqFile.estimate_project_id,
                Project.project_name,
                Project.project_code,
                EstimateBoqFile.created_at
            ).join(
                EstimateBoqProject, 
                EstimateBoqFile.estimate_project_id == EstimateBoqProject.estimate_project_id
            ).join(
                Project, EstimateBoqProject.project_id == Project.project_id
            ).filter(
                EstimateBoqFile.boq_id == boq_id
            ).first()
            
            if result:
                return {
                    'boq_id': result.boq_id,
                    'file_name': result.file_name,
                    'estimate_project_id': result.estimate_project_id,
                    'project_name': result.project_name,
                    'project_code': result.project_code,
                    'created_at': result.created_at.isoformat() if result.created_at else None
                }
            return None
        finally:
            session.close()
