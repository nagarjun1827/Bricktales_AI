from sqlalchemy import (
    Column, Integer, Text, String, TIMESTAMP, ForeignKey, DECIMAL, Computed
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
    StoreBoqItem model with computed columns.
    
    The supply_amount, labour_amount, and total_amount columns are GENERATED STORED
    columns in PostgreSQL. They are declared here so SQLAlchemy can READ them,
    but SQLAlchemy will NOT try to INSERT into them due to the Computed() declaration.
    """
    __tablename__ = "store_boq_items"

    item_id = Column(Integer, primary_key=True)
    boq_id = Column(Integer, ForeignKey("store_boq_files.boq_id"), nullable=False)
    item_code = Column(String(50))
    item_description = Column(Text, nullable=False)
    unit_of_measurement = Column(String(20), nullable=False)
    quantity = Column(DECIMAL(18, 3), nullable=False)
    
    # Rate columns - NOT NULL with defaults
    supply_unit_rate = Column(
        DECIMAL(18, 2), 
        nullable=False, 
        default=0.0, 
        server_default='0.0'
    )
    labour_unit_rate = Column(
        DECIMAL(18, 2), 
        nullable=False, 
        default=0.0, 
        server_default='0.0'
    )
    
    # Computed columns - SQLAlchemy can READ these but won't INSERT into them
    # The key is using Computed() which tells SQLAlchemy these are server-side generated
    supply_amount = Column(
        DECIMAL(18, 2),
        Computed("quantity * supply_unit_rate", persisted=True)
    )
    labour_amount = Column(
        DECIMAL(18, 2),
        Computed("quantity * labour_unit_rate", persisted=True)
    )
    total_amount = Column(
        DECIMAL(18, 2),
        Computed("quantity * supply_unit_rate + quantity * labour_unit_rate", persisted=True)
    )
    
    location_id = Column(Integer, ForeignKey("store_boq_locations.location_id"))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    description_embedding = Column(Vector(768))
    embedding_generated_at = Column(TIMESTAMP(timezone=True))

    file = relationship("StoreBoqFile", back_populates="items")
    location = relationship("StoreBoqLocation", back_populates="items")