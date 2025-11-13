from sqlalchemy import (
    Column, Integer, Text, String, TIMESTAMP, ForeignKey, DECIMAL
)
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from models.base import Base


class StoreBoqProject(Base):
    __tablename__ = "store_boq_projects"

    store_project_id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    store_project_name = Column(Text)
    store_project_code = Column(String(50))
    total_project_value = Column(DECIMAL(18, 2))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="store_boq_projects")
    locations = relationship("StoreBoqLocation", back_populates="store_project")
    files = relationship("StoreBoqFile", back_populates="store_project")


class StoreBoqLocation(Base):
    __tablename__ = "store_boq_locations"

    location_id = Column(Integer, primary_key=True)
    store_project_id = Column(Integer, ForeignKey("store_boq_projects.store_project_id"), nullable=False)
    location_name = Column(Text)
    address = Column(Text)
    latitude = Column(DECIMAL(9, 6))
    longitude = Column(DECIMAL(9, 6))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    store_project = relationship("StoreBoqProject", back_populates="locations")
    items = relationship("StoreBoqItem", back_populates="location")


class StoreBoqFile(Base):
    __tablename__ = "store_boq_files"

    boq_id = Column(Integer, primary_key=True)
    store_project_id = Column(Integer, ForeignKey("store_boq_projects.store_project_id"), nullable=False)
    file_name = Column(Text, nullable=False)
    file_path = Column(Text)
    file_type = Column(String(20))
    version = Column(Integer, default=1)
    is_active = Column(String, default=True)
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    store_project = relationship("StoreBoqProject", back_populates="files")
    items = relationship("StoreBoqItem", back_populates="file")


class StoreBoqItem(Base):
    """
    StoreBoqItem model - COMPUTED COLUMNS NOT DECLARED HERE.
    
    The columns supply_amount, labour_amount, and total_amount exist in the database
    as GENERATED STORED columns, but we DON'T declare them in SQLAlchemy.
    
    This prevents SQLAlchemy from trying to insert values into them.
    If you need to read these values, query them explicitly in your SQL.
    """
    __tablename__ = "store_boq_items"

    item_id = Column(Integer, primary_key=True)
    boq_id = Column(Integer, ForeignKey("store_boq_files.boq_id"), nullable=False)
    item_code = Column(String(50))
    item_description = Column(Text, nullable=False)
    unit_of_measurement = Column(String(20), nullable=False)
    quantity = Column(DECIMAL(18, 3), nullable=False)
    supply_unit_rate = Column(DECIMAL(18, 2), nullable=False, default=0.0, server_default='0.0')
    labour_unit_rate = Column(DECIMAL(18, 2), nullable=False, default=0.0, server_default='0.0')
    
    # NOTE: supply_amount, labour_amount, total_amount are NOT declared here
    # They exist in the database as GENERATED columns
    # PostgreSQL will calculate them automatically on INSERT
    
    location_id = Column(Integer, ForeignKey("store_boq_locations.location_id"))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    description_embedding = Column(Vector(768))
    embedding_generated_at = Column(TIMESTAMP(timezone=True))

    file = relationship("StoreBoqFile", back_populates="items")
    location = relationship("StoreBoqLocation", back_populates="items")