# models/tender_models.py
from sqlalchemy import (
    Column, Integer, Text, String, Boolean, TIMESTAMP, ForeignKey, ARRAY
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from models.base import Base


class TenderProject(Base):
    __tablename__ = "tender_projects"

    tender_id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    tender_number = Column(String(100))
    tender_date = Column(TIMESTAMP(timezone=True))
    submission_deadline = Column(TIMESTAMP(timezone=True))
    tender_status = Column(String(50))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="tender_projects")
    tender_files = relationship("TenderFile", back_populates="tender_project")


class TenderFile(Base):
    __tablename__ = "tender_files"

    tender_file_id = Column(Integer, primary_key=True)
    tender_id = Column(Integer, ForeignKey("tender_projects.tender_id"), nullable=False)
    file_name = Column(Text, nullable=False)
    file_path = Column(Text)
    file_type = Column(String(20))
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    summary = Column(Text)
    simple_summary = Column(Text)
    bm25_corpus = Column(JSONB)
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    tender_project = relationship("TenderProject", back_populates="tender_files")
    chunks = relationship("TenderChunk", back_populates="tender_file")


class TenderChunk(Base):
    __tablename__ = "tender_chunks"

    id = Column(Integer, primary_key=True)
    tender_file_id = Column(Integer, ForeignKey("tender_files.tender_file_id", ondelete="CASCADE"))
    chunk_index = Column(Integer)
    chunk_text = Column(Text)
    chunk_metadata = Column(JSONB)
    dense_embedding = Column(Vector(768))
    sparse_embedding = Column(JSONB)
    bm25_tokens = Column(ARRAY(Text))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    tender_file = relationship("TenderFile", back_populates="chunks")