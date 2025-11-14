from sqlalchemy import (
    Column, Integer, Text, String, Boolean,TIMESTAMP, ForeignKey, DECIMAL
)
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from models.base import Base


class EstimateBoqProject(Base):
    __tablename__ = "estimate_boq_projects"

    estimate_project_id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    estimate_project_name = Column(Text)
    estimate_project_code = Column(String(50))
    estimation_status = Column(String(50))
    estimated_value = Column(DECIMAL(18, 2))
    estimated_by = Column(Text)
    estimated_at = Column(TIMESTAMP(timezone=True))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="estimate_boq_projects")
    locations = relationship("EstimateBoqLocation", back_populates="estimate_project")
    files = relationship("EstimateBoqFile", back_populates="estimate_project")


class EstimateBoqLocation(Base):
    __tablename__ = "estimate_boq_locations"

    location_id = Column(Integer, primary_key=True)
    estimate_project_id = Column(Integer, ForeignKey("estimate_boq_projects.estimate_project_id"), nullable=False)
    location_name = Column(Text)
    address = Column(Text)
    latitude = Column(DECIMAL(9, 6))
    longitude = Column(DECIMAL(9, 6))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    estimate_project = relationship("EstimateBoqProject", back_populates="locations")
    items = relationship("EstimateBoqItem", back_populates="location")


class EstimateBoqFile(Base):
    __tablename__ = "estimate_boq_files"

    boq_id = Column(Integer, primary_key=True)
    estimate_project_id = Column(Integer, ForeignKey("estimate_boq_projects.estimate_project_id"), nullable=False)
    file_name = Column(Text, nullable=False)
    file_path = Column(Text)
    file_type = Column(String(20))
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    estimate_project = relationship("EstimateBoqProject", back_populates="files")
    items = relationship("EstimateBoqItem", back_populates="file")


class EstimateBoqItem(Base):
    __tablename__ = "estimate_boq_items"

    item_id = Column(Integer, primary_key=True)
    boq_id = Column(Integer, ForeignKey("estimate_boq_files.boq_id"), nullable=False)
    item_code = Column(String(50))
    item_description = Column(Text, nullable=False)
    unit_of_measurement = Column(String(20), nullable=False)
    quantity = Column(DECIMAL(18, 3), nullable=False)
    location_id = Column(Integer, ForeignKey("estimate_boq_locations.location_id"))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    description_embedding = Column(Vector(768))
    embedding_generated_at = Column(TIMESTAMP(timezone=True))

    file = relationship("EstimateBoqFile", back_populates="items")
    location = relationship("EstimateBoqLocation", back_populates="items")
