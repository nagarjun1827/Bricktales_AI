"""
Repository for store BOQ database operations.
"""
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict
from models.domain import ProjectInfo, LocationInfo, BOQFileInfo, BOQItem
from core.settings import settings


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
        """Insert location."""
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
        """Insert BOQ file."""
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
        
        print(f"   Inserting items with:")
        print(f"   - Total Supply Amount: ₹{total_supply:,.2f}")
        print(f"   - Total Labour Amount: ₹{total_labour:,.2f}")
        print(f"   - Total Amount:        ₹{total_amount:,.2f}")
            
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

    def get_boq_totals(self, boq_id: int) -> Dict[str, float]:
        """
        Get BOQ totals from database.
        
        The amounts are calculated by the database using GENERATED columns:
        - supply_amount = quantity * supply_unit_rate
        - labour_amount = quantity * labour_unit_rate
        - total_amount = supply_amount + labour_amount
        """
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
            
            print(f"   Retrieved totals from database:")
            print(f"   - Item Count:    {totals['item_count']}")
            print(f"   - Supply Amount: ₹{totals['total_supply']:,.2f}")
            print(f"   - Labour Amount: ₹{totals['total_labour']:,.2f}")
            print(f"   - Total Amount:  ₹{totals['total_amount']:,.2f}")
            
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