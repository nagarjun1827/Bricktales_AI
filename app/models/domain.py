from dataclasses import dataclass
from typing import Optional
from datetime import date

@dataclass
class ProjectInfo:
    project_name: str
    project_code: str
    client_id: Optional[int] = None
    client_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    version: int = 1
    created_by: str = "system"

@dataclass
class LocationInfo:
    project_id: int
    location_name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_by: str = "system"

@dataclass
class BOQFileInfo:
    project_id: int
    file_name: str
    file_path: str
    file_type: str = "xlsx"
    version: int = 1
    is_active: bool = True
    created_by: str = "system"

@dataclass
class BOQItem:
    boq_id: int
    item_description: str = ""
    item_code: Optional[str] = None
    unit_of_measurement: str = ""
    quantity: float = 0.0
    supply_unit_rate: float = 0.0
    labour_unit_rate: float = 0.0
    location_id: Optional[int] = None
    created_by: str = "system"
    supply_amount: float = 0.0
    labour_amount: float = 0.0
    total_amount: float = 0.0

    def calculate_amounts(self) -> None:
        self.supply_amount = self.supply_unit_rate * self.quantity
        self.labour_amount = self.labour_unit_rate * self.quantity
        self.total_amount = self.supply_amount + self.labour_amount