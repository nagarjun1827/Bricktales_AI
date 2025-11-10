import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional
from models.estimate_boq import TBEProjectInfo, EstimateBOQProjectInfo, TBELocationInfo, TBEBOQFileInfo, TBEBOQItem
from core.settings import settings

class TBEBOQRepository:
    """Repository for To-Be-Estimated BOQ operations"""
    
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

    def insert_project(self, project_info: TBEProjectInfo) -> int:
        """Insert project record"""
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

    def insert_estimate_boq_project(self, estimate_project_info: EstimateBOQProjectInfo) -> int:
        """Insert estimate BOQ project."""
        query = """
            INSERT INTO estimate_boq_projects (
                project_id, estimate_project_name, estimate_project_code,
                estimation_status, estimated_value, estimated_by,
                estimated_at, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING estimate_project_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                estimate_project_info.project_id,
                estimate_project_info.estimate_project_name,
                estimate_project_info.estimate_project_code,
                estimate_project_info.estimation_status,
                estimate_project_info.estimated_value,
                estimate_project_info.estimated_by,
                estimate_project_info.estimated_at,
                estimate_project_info.created_by,
            ))
            estimate_project_id = cur.fetchone()[0]
            conn.commit()
            return estimate_project_id

    def insert_location(self, location_info: TBELocationInfo) -> int:
        """Insert location record"""
        query = """
            INSERT INTO estimate_boq_locations (
                estimate_project_id, location_name, address, 
                latitude, longitude, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING location_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                location_info.estimate_project_id,
                location_info.location_name,
                location_info.address,
                location_info.latitude,
                location_info.longitude,
                location_info.created_by,
            ))
            location_id = cur.fetchone()[0]
            conn.commit()
            return location_id

    def insert_tbe_boq_file(self, file_info: TBEBOQFileInfo) -> int:
        """Insert to-be-estimated BOQ file record"""
        query = """
            INSERT INTO estimate_boq_files (
                estimate_project_id, file_name, file_path, file_type,
                version, is_active, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING boq_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                file_info.estimate_project_id,
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

    def insert_tbe_items_batch(self, items: List[TBEBOQItem]) -> None:
        """Batch insert to-be-estimated BOQ items (no pricing)"""
        if not items:
            print("   ⚠️  No items to insert")
            return
        
        print(f"   Preparing to insert {len(items)} items...")
        
        items_data = [
            (
                item.boq_id,
                item.item_code,
                item.item_description,
                item.unit_of_measurement,
                item.quantity,
                item.location_id,
                item.created_by,
            )
            for item in items
        ]
        
        print(f"   Data prepared for {len(items_data)} items")
        
        query = """
            INSERT INTO estimate_boq_items (
                boq_id, item_code, item_description, 
                unit_of_measurement, quantity, 
                location_id, created_by
            ) VALUES %s
        """
        
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                print("   Executing batch insert...")
                execute_values(cur, query, items_data)
                print(f"   Committing {cur.rowcount} rows...")
                conn.commit()
                print(f"   ✓ Successfully inserted {cur.rowcount} items")
        except Exception as e:
            print(f"   ✗ Insert failed: {e}")
            raise

    def get_tbe_boq_summary(self, boq_id: int) -> Dict[str, int]:
        """Get summary of to-be-estimated BOQ"""
        query = """
            SELECT COUNT(*) as item_count
            FROM estimate_boq_items
            WHERE boq_id = %s
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            result = cur.fetchone()
            
            return {
                "item_count": result[0] if result else 0
            }

    def get_tbe_items_by_boq(self, boq_id: int, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get TBE items for a BOQ with pagination"""
        query = """
            SELECT 
                item_id, boq_id, item_code, item_description,
                unit_of_measurement, quantity, location_id,
                created_by, created_at, updated_at
            FROM estimate_boq_items
            WHERE boq_id = %s
            ORDER BY item_id
            LIMIT %s OFFSET %s
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id, limit, offset))
            rows = cur.fetchall()
            
            items = []
            for row in rows:
                items.append({
                    "item_id": row[0],
                    "boq_id": row[1],
                    "item_code": row[2],
                    "item_description": row[3],
                    "unit_of_measurement": row[4],
                    "quantity": float(row[5]),
                    "location_id": row[6],
                    "created_by": row[7],
                    "created_at": row[8].isoformat() if row[8] else None,
                    "updated_at": row[9].isoformat() if row[9] else None,
                })
            
            return items
        
    def delete_boq_by_id(self, boq_id: int) -> Dict[str, Any]:
        """
        Delete Estimate BOQ file and all related data.
        Returns count of deleted records from each table.
        """
        try:
            with self._get_connection() as conn, conn.cursor() as cur:
                deleted_counts = {}
                
                # Get BOQ info before deletion
                cur.execute("""
                    SELECT ebf.boq_id, ebf.estimate_project_id, ebf.file_name
                    FROM estimate_boq_files ebf
                    WHERE ebf.boq_id = %s
                """, (boq_id,))
                
                boq_info = cur.fetchone()
                if not boq_info:
                    return {
                        'success': False,
                        'error': f'Estimate BOQ file with ID {boq_id} not found'
                    }
                
                estimate_project_id = boq_info[1]
                file_name = boq_info[2]
                
                print(f"🗑️  Deleting Estimate BOQ: {file_name} (ID: {boq_id})")
                
                # Step 1: Delete BOQ items
                cur.execute("""
                    DELETE FROM estimate_boq_items 
                    WHERE boq_id = %s
                """, (boq_id,))
                deleted_counts['boq_items'] = cur.rowcount
                print(f"   ✓ Deleted {cur.rowcount} BOQ items")
                
                # Step 2: Delete BOQ file record
                cur.execute("""
                    DELETE FROM estimate_boq_files 
                    WHERE boq_id = %s
                """, (boq_id,))
                deleted_counts['boq_file'] = cur.rowcount
                print(f"   ✓ Deleted BOQ file record")
                
                # Step 3: Check if estimate project has other BOQ files
                cur.execute("""
                    SELECT COUNT(*) FROM estimate_boq_files 
                    WHERE estimate_project_id = %s
                """, (estimate_project_id,))
                
                remaining_boqs = cur.fetchone()[0]
                
                if remaining_boqs == 0:
                    # Delete estimate project and related data if no other BOQs exist
                    print(f"   No other BOQs found for estimate project {estimate_project_id}")
                    
                    # Get project_id before deleting estimate project
                    cur.execute("""
                        SELECT project_id FROM estimate_boq_projects 
                        WHERE estimate_project_id = %s
                    """, (estimate_project_id,))
                    
                    project_result = cur.fetchone()
                    if project_result:
                        project_id = project_result[0]
                        
                        # Delete locations
                        cur.execute("""
                            DELETE FROM estimate_boq_locations 
                            WHERE estimate_project_id = %s
                        """, (estimate_project_id,))
                        deleted_counts['locations'] = cur.rowcount
                        print(f"   ✓ Deleted {cur.rowcount} locations")
                        
                        # Delete estimate project
                        cur.execute("""
                            DELETE FROM estimate_boq_projects 
                            WHERE estimate_project_id = %s
                        """, (estimate_project_id,))
                        deleted_counts['estimate_project'] = cur.rowcount
                        print(f"   ✓ Deleted estimate project")
                        
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
                            print(f"   ✓ Deleted main project")
                        else:
                            print(f"   ℹ️  Main project retained (has other references)")
                            deleted_counts['project'] = 0
                    else:
                        deleted_counts['locations'] = 0
                        deleted_counts['estimate_project'] = 0
                        deleted_counts['project'] = 0
                else:
                    print(f"   ℹ️  Estimate project retained ({remaining_boqs} other BOQ(s) exist)")
                    deleted_counts['locations'] = 0
                    deleted_counts['estimate_project'] = 0
                    deleted_counts['project'] = 0
                
                conn.commit()
                
                print(f"✓ Successfully deleted Estimate BOQ {boq_id}")
                
                return {
                    'success': True,
                    'boq_id': boq_id,
                    'deleted_counts': deleted_counts,
                    'message': f'Estimate BOQ {boq_id} and related data deleted successfully'
                }
                
        except Exception as e:
            print(f"✗ Error deleting Estimate BOQ: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def get_boq_info(self, boq_id: int) -> Optional[Dict]:
        """Get Estimate BOQ file information."""
        query = """
            SELECT 
                ebf.boq_id,
                ebf.file_name,
                ebf.estimate_project_id,
                p.project_name,
                p.project_code,
                ebf.created_at
            FROM estimate_boq_files ebf
            JOIN estimate_boq_projects ebp ON ebf.estimate_project_id = ebp.estimate_project_id
            JOIN projects p ON ebp.project_id = p.project_id
            WHERE ebf.boq_id = %s
        """
        
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (boq_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    'boq_id': row[0],
                    'file_name': row[1],
                    'estimate_project_id': row[2],
                    'project_name': row[3],
                    'project_code': row[4],
                    'created_at': row[5].isoformat() if row[5] else None
                }
            return None