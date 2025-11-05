import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict
from app.models.domain import ProjectInfo, LocationInfo, BOQFileInfo, BOQItem
from app.core.config import settings

class BoQRepository:
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

    def insert_project(self, project_info: ProjectInfo) -> int:
        query = """
            INSERT INTO projects (
                project_name, project_code, client_id, client_name,
                start_date, end_date, version, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING project_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                project_info.project_name, project_info.project_code,
                project_info.client_id, project_info.client_name,
                project_info.start_date, project_info.end_date,
                project_info.version, project_info.created_by,
            ))
            project_id = cur.fetchone()[0]
            conn.commit()
            return project_id

    def insert_location(self, location_info: LocationInfo) -> int:
        query = """
            INSERT INTO locations (
                project_id, location_name, address, latitude, longitude, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING location_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                location_info.project_id, location_info.location_name,
                location_info.address, location_info.latitude,
                location_info.longitude, location_info.created_by,
            ))
            location_id = cur.fetchone()[0]
            conn.commit()
            return location_id

    def insert_boq_file(self, file_info: BOQFileInfo) -> int:
        query = """
            INSERT INTO store_boq_files (
                project_id, file_name, file_path, file_type,
                version, is_active, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING boq_id
        """
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, (
                file_info.project_id, file_info.file_name,
                file_info.file_path, file_info.file_type,
                file_info.version, file_info.is_active,
                file_info.created_by,
            ))
            boq_id = cur.fetchone()[0]
            conn.commit()
            return boq_id

    def insert_boq_items_batch(self, items: List[BOQItem]) -> None:
        if not items:
            return
        items_data = [
            (
                item.boq_id, item.item_code, item.item_description,
                item.unit_of_measurement, item.quantity,
                item.supply_unit_rate, item.labour_unit_rate,
                item.location_id, item.created_by,
            )
            for item in items
        ]
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

    def get_boq_totals(self, boq_id: int) -> Dict[str, float]:
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
            return {
                "item_count": result[0],
                "total_supply": float(result[1]),
                "total_labour": float(result[2]),
                "total_amount": float(result[3]),
            }