from dataclasses import dataclass
from typing import Optional

@dataclass
class TBEProjectInfo:
    project_name: str
    project_code: str
    project_type: str = "boq"
    client_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    version: int = 1
    created_by: str = "system"


@dataclass
class EstimateBOQProjectInfo:
    """Estimate BOQ project information."""
    project_id: int
    estimate_project_name: str
    estimate_project_code: str
    estimation_status: str = "pending"
    estimated_value: Optional[float] = None
    estimated_by: Optional[str] = None
    estimated_at: Optional[str] = None
    created_by: str = "system"


@dataclass
class TBELocationInfo:
    estimate_project_id: int
    location_name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_by: str = "system"


@dataclass
class TBEBOQFileInfo:
    estimate_project_id: int
    file_name: str
    file_path: str
    file_type: str = "xlsx"
    version: int = 1
    is_active: bool = True
    created_by: str = "system"


@dataclass
class TBEBOQItem:
    boq_id: int
    item_description: str = ""
    item_code: Optional[str] = None
    unit_of_measurement: str = "Each"
    quantity: float = 0.0
    location_id: Optional[int] = None
    created_by: str = "system"