from sqlalchemy import Column, Integer, Text, String, TIMESTAMP, Date, ForeignKey, DECIMAL
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from models.base import Base


class TenderProject(Base):
    __tablename__ = "tender_projects"

    tender_id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.project_id"), nullable=False)
    tender_number = Column(String(100))
    tender_date = Column(Date)
    submission_deadline = Column(TIMESTAMP(timezone=True))
    tender_status = Column(String(50))
    tender_value = Column(DECIMAL(18, 2))
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="tender_projects")
    tender_files = relationship("TenderFile", back_populates="tender")


class TenderFile(Base):
    __tablename__ = "tender_files"

    tender_file_id = Column(Integer, primary_key=True)
    tender_id = Column(Integer, ForeignKey("tender_projects.tender_id"), nullable=False)
    file_name = Column(Text, nullable=False)
    file_path = Column(Text)
    file_type = Column(String(20))
    version = Column(Integer, default=1)
    is_active = Column(String, default=True)
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    tender = relationship("TenderProject", back_populates="tender_files")
