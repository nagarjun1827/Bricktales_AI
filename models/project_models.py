from sqlalchemy import (
    Column, Integer, String, Text, SmallInteger, TIMESTAMP, Date, ForeignKey
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from models.base import Base


class Company(Base):
    __tablename__ = "company"

    cmp_id = Column(Integer, primary_key=True)
    cmp_name = Column(String(145), nullable=False)
    cmp_addr = Column(String(45))
    fullname = Column(String(145))
    cmp_phone = Column(String(45))
    username = Column(String(45))
    password = Column(String(145))
    access_otp = Column(Integer)
    cmp_status = Column(SmallInteger)
    isenable = Column(SmallInteger)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    projects = relationship("Project", back_populates="client")


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(Integer, primary_key=True)
    project_name = Column(Text)
    project_code = Column(String(50))
    project_type = Column(String(10), nullable=False)
    client_id = Column(Integer, ForeignKey("company.cmp_id"))
    client_name = Column(Text)
    start_date = Column(Date)
    end_date = Column(Date)
    version = Column(Integer, default=1)
    created_by = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_by = Column(Text)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    client = relationship("Company", back_populates="projects")
    tender_projects = relationship("TenderProject", back_populates="project")
    store_boq_projects = relationship("StoreBoqProject", back_populates="project")
    estimate_boq_projects = relationship("EstimateBoqProject", back_populates="project")
