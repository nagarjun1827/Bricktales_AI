"""
Repository for store BOQ database operations.
"""
import psycopg2
import logging
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional
from models.store_boq_models import ProjectInfo, StoreBOQProjectInfo, LocationInfo, BOQFileInfo, BOQItem
from core.settings import settings

logger = logging.getLogger(__name__)


class BOQItemWithEmbedding:
    """BOQ item with embedding data."""
    def __init__(self, item_id, boq_id, item_code, item_description, 
                 unit_of_measurement, quantity, supply_unit_rate, 
                 supply_amount, labour_unit_rate, labour_amount, 
                 total_amount, location_id, created_by, created_at,
                 updated_by, updated_at, description_embedding=None, 
                 embedding_generated_at=None):
        self.item_id = item_id
        self.boq_id = boq_id
        self.item_code = item_code
        self.item_description = item_description
        self.unit_of_measurement = unit_of_measurement
        self.quantity = quantity
        self.supply_unit_rate = supply_unit_rate
        self.supply_amount = supply_amount
        self.labour_unit_rate = labour_unit_rate
        self.labour_amount = labour_amount
        self.total_amount = total_amount
        self.location_id = location_id
        self.created_by = created_by
        self.created_at = created_at
        self.updated_by = updated_by
        self.updated_at = updated_at
        self.description_embedding = description_embedding
        self.embedding_generated_at = embedding_generated_at


class StoreBOQRepository:
    """Repository for store BOQ database operations."""
    
    def __init__(self):
        self.db_config = {
            "host": settings.DB_HOST,
            "database": settings.DB_NAME,
            "user": settings.DB_USER,
            "password": settings.DB_PASSWORD,
            "port": settings.DB_PORT,
        }

    def _get_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def insert_project(self, project_info: ProjectInfo) -> int:
        """Insert project."""
        query = """
            INSERT INTO projects (
                project_name, project_code, project_type, client_name,
                start_date, end_date, version, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING project_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                project_info.project_name, 
                project_info.project_code,
                project_info.project_type,
                project_info.client_name,
                project_info.start_date, 
                project_info.end_date,
                project_info.version, 
                project_info.created_by,
            ))
            project_id = cur.fetchone()[0]
            conn.commit()
            return project_id

    def insert_store_boq_project(self, store_project_info: StoreBOQProjectInfo) -> int:
        """Insert store BOQ project."""
        query = """
            INSERT INTO store_boq_projects (
                project_id, store_project_name, store_project_code,
                total_project_value, created_by
            ) VALUES (%s, %s, %s, %s, %s)
            RETURNING store_project_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                store_project_info.project_id,
                store_project_info.store_project_name,
                store_project_info.store_project_code,
                store_project_info.total_project_value,
                store_project_info.created_by,
            ))
            store_project_id = cur.fetchone()[0]
            conn.commit()
            return store_project_id

    def insert_location(self, location_info: LocationInfo) -> int:
        """Insert location."""
        query = """
            INSERT INTO store_boq_locations (
                store_project_id, location_name, address, latitude, longitude, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING location_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                location_info.store_project_id,
                location_info.location_name,
                location_info.address, 
                location_info.latitude,
                location_info.longitude, 
                location_info.created_by,
            ))
            location_id = cur.fetchone()[0]
            conn.commit()
            return location_id

    def insert_boq_file(self, file_info: BOQFileInfo) -> int:
        """Insert BOQ file."""
        query = """
            INSERT INTO store_boq_files (
                store_project_id, file_name, file_path, file_type,
                version, is_active, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING boq_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                file_info.store_project_id,
                file_info.file_name,
                file_info.file_path, 
                file_info.file_type,
                file_info.version, 
                file_info.is_active,
                file_info.created_by,
            ))
            boq_id = cur.fetchone()[0]
            conn.commit()
            return boq_id

    def insert_boq_items_batch(self, items: List[BOQItem]) -> None:
        """
        Batch insert BOQ items.
        
        Note: supply_amount, labour_amount, and total_amount are GENERATED columns
        in the database (calculated automatically), so we don't insert them.
        """
        if not items:
            return
        
        # Log total amounts for verification
        total_supply = sum(item.supply_amount for item in items)
        total_labour = sum(item.labour_amount for item in items)
        total_amount = sum(item.total_amount for item in items)
        
        logger.info(f"Inserting items - Supply: ₹{total_supply:,.2f}, Labour: ₹{total_labour:,.2f}, Total: ₹{total_amount:,.2f}")
            
        items_data = [
            (
                item.boq_id, 
                item.item_code, 
                item.item_description,
                item.unit_of_measurement, 
                item.quantity,
                item.supply_unit_rate, 
                item.labour_unit_rate,
                item.location_id, 
                item.created_by,
            )
            for item in items
        ]
        
        # Note: We don't insert supply_amount, labour_amount, total_amount
        # because they are GENERATED ALWAYS AS columns in the database
        query = """
            INSERT INTO store_boq_items (
                boq_id, item_code, item_description, unit_of_measurement,
                quantity, supply_unit_rate, labour_unit_rate,
                location_id, created_by
            ) VALUES %s
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            execute_values(cur, query, items_data)
            conn.commit()

    def get_boq_totals(self, boq_id: int, items: List[BOQItem] = None) -> Dict[str, float]:
        """
        Get BOQ totals from database or calculate from items list.
        
        Args:
            boq_id: BOQ file ID
            items: Optional list of BOQItem objects to calculate from
            
        Returns:
            Dictionary with item_count, total_supply, total_labour, total_amount
        """
        if items:
            # Calculate from provided items (fallback when database hasn't computed yet)
            totals = {
                "item_count": len(items),
                "total_supply": sum(item.supply_amount for item in items),
                "total_labour": sum(item.labour_amount for item in items),
                "total_amount": sum(item.total_amount for item in items),
            }
            
            logger.info(f"Calculated totals from {len(items)} items")
            logger.info(f"Item Count: {totals['item_count']}")
            logger.info(f"Supply: ₹{totals['total_supply']:,.2f}")
            logger.info(f"Labour: ₹{totals['total_labour']:,.2f}")
            logger.info(f"Total: ₹{totals['total_amount']:,.2f}")
            
            return totals
        
        # Query from database (when items not provided)
        query = """
            SELECT 
                COUNT(*) as item_count,
                COALESCE(SUM(supply_amount), 0) as total_supply,
                COALESCE(SUM(labour_amount), 0) as total_labour,
                COALESCE(SUM(total_amount), 0) as total_amount
            FROM store_boq_items
            WHERE boq_id = %s
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            result = cur.fetchone()
            
            totals = {
                "item_count": result[0],
                "total_supply": float(result[1]),
                "total_labour": float(result[2]),
                "total_amount": float(result[3]),
            }
            
            logger.info("Retrieved totals from database")
            logger.info(f"Item Count: {totals['item_count']}")
            logger.info(f"Supply: ₹{totals['total_supply']:,.2f}")
            logger.info(f"Labour: ₹{totals['total_labour']:,.2f}")
            logger.info(f"Total: ₹{totals['total_amount']:,.2f}")
            
            return totals
    
    # === Embedding Methods ===
    
    def ensure_vector_extension(self):
        """Ensure pgvector extension."""
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    
    def get_items_without_embeddings(self, boq_id: int) -> List[BOQItemWithEmbedding]:
        """Get items without embeddings."""
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
            return [BOQItemWithEmbedding(*row) for row in rows]
    
    def update_embeddings_batch(self, item_ids: List[int], embeddings: List[List[float]]) -> int:
        """Update embeddings."""
        if not item_ids or not embeddings:
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
        
    def delete_boq_by_id(self, boq_id: int) -> Dict[str, Any]:
        """
        Delete BOQ file and all related data.
        Returns count of deleted records from each table.
        """
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                deleted_counts = {}
                
                # Get BOQ info before deletion
                cur.execute("""
                    SELECT sbf.boq_id, sbf.store_project_id, sbf.file_name
                    FROM store_boq_files sbf
                    WHERE sbf.boq_id = %s
                """, (boq_id,))
                
                boq_info = cur.fetchone()
                if not boq_info:
                    return {
                        'success': False,
                        'error': f'BOQ file with ID {boq_id} not found'
                    }
                
                store_project_id = boq_info[1]
                file_name = boq_info[2]
                
                logger.info(f"Deleting Store BOQ: {file_name} (ID: {boq_id})")
                
                # Step 1: Delete BOQ items
                cur.execute("""
                    DELETE FROM store_boq_items 
                    WHERE boq_id = %s
                """, (boq_id,))
                deleted_counts['boq_items'] = cur.rowcount
                logger.info(f"Deleted {cur.rowcount} BOQ items")
                
                # Step 2: Delete BOQ file record
                cur.execute("""
                    DELETE FROM store_boq_files 
                    WHERE boq_id = %s
                """, (boq_id,))
                deleted_counts['boq_file'] = cur.rowcount
                logger.info("Deleted BOQ file record")
                
                # Step 3: Check if store project has other BOQ files
                cur.execute("""
                    SELECT COUNT(*) FROM store_boq_files 
                    WHERE store_project_id = %s
                """, (store_project_id,))
                
                remaining_boqs = cur.fetchone()[0]
                
                if remaining_boqs == 0:
                    # Delete store project and related data if no other BOQs exist
                    logger.info(f"No other BOQs found for store project {store_project_id}")
                    
                    # Get project_id before deleting store project
                    cur.execute("""
                        SELECT project_id FROM store_boq_projects 
                        WHERE store_project_id = %s
                    """, (store_project_id,))
                    
                    project_result = cur.fetchone()
                    if project_result:
                        project_id = project_result[0]
                        
                        # Delete locations
                        cur.execute("""
                            DELETE FROM store_boq_locations 
                            WHERE store_project_id = %s
                        """, (store_project_id,))
                        deleted_counts['locations'] = cur.rowcount
                        logger.info(f"Deleted {cur.rowcount} locations")
                        
                        # Delete store project
                        cur.execute("""
                            DELETE FROM store_boq_projects 
                            WHERE store_project_id = %s
                        """, (store_project_id,))
                        deleted_counts['store_project'] = cur.rowcount
                        logger.info("Deleted store project")
                        
                        # Check if project has other store/estimate projects
                        cur.execute("""
                            SELECT 
                                (SELECT COUNT(*) FROM store_boq_projects WHERE project_id = %s) +
                                (SELECT COUNT(*) FROM estimate_boq_projects WHERE project_id = %s)
                        """, (project_id, project_id))
                        
                        remaining_projects = cur.fetchone()[0]
                        
                        if remaining_projects == 0:
                            # Delete main project if no other references
                            cur.execute("""
                                DELETE FROM projects 
                                WHERE project_id = %s
                            """, (project_id,))
                            deleted_counts['project'] = cur.rowcount
                            logger.info("Deleted main project")
                        else:
                            logger.info("Main project retained (has other references)")
                            deleted_counts['project'] = 0
                    else:
                        deleted_counts['locations'] = 0
                        deleted_counts['store_project'] = 0
                        deleted_counts['project'] = 0
                else:
                    logger.info(f"Store project retained ({remaining_boqs} other BOQ(s) exist)")
                    deleted_counts['locations'] = 0
                    deleted_counts['store_project'] = 0
                    deleted_counts['project'] = 0
                
                conn.commit()
                
                logger.info(f"Successfully deleted Store BOQ {boq_id}")
                
                return {
                    'success': True,
                    'boq_id': boq_id,
                    'deleted_counts': deleted_counts,
                    'message': f'Store BOQ {boq_id} and related data deleted successfully'
                }
                
        except Exception as e:
            logger.error(f"Error deleting Store BOQ: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get BOQ file information."""
        query = """
            SELECT 
                sbf.boq_id,
                sbf.file_name,
                sbf.store_project_id,
                p.project_name,
                p.project_code,
                sbf.created_at
            FROM store_boq_files sbf
            JOIN store_boq_projects sbp ON sbf.store_project_id = sbp.store_project_id
            JOIN projects p ON sbp.project_id = p.project_id
            WHERE sbf.boq_id = %s
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    'boq_id': row[0],
                    'file_name': row[1],
                    'store_project_id': row[2],
                    'project_name': row[3],
                    'project_code': row[4],
                    'created_at': row[5].isoformat() if row[5] else None
                }
            return None