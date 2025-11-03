"""
Domain models for BOQ entities.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class ProjectInfo:
    """Project information."""
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
    """Location information."""
    project_id: int
    location_name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_by: str = "system"


@dataclass
class BOQFileInfo:
    """BOQ file information."""
    project_id: int
    file_name: str
    file_path: str
    file_type: str = "xlsx"
    version: int = 1
    is_active: bool = True
    created_by: str = "system"


@dataclass
class BOQItem:
    """BOQ item with rates and amounts."""
    boq_id: int
    item_description: str
    item_code: Optional[str] = None
    unit_of_measurement: str = "Each"
    quantity: float = 0.0
    supply_unit_rate: float = 0.0
    labour_unit_rate: float = 0.0
    location_id: Optional[int] = None
    created_by: str = "system"
    
    # Calculated fields
    supply_amount: float = field(init=False, default=0.0)
    labour_amount: float = field(init=False, default=0.0)
    total_amount: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Calculate amounts after initialization."""
        self.calculate_amounts()

    def calculate_amounts(self) -> None:
        """Calculate supply, labour, and total amounts."""
        self.supply_amount = round(self.supply_unit_rate * self.quantity, 2)
        self.labour_amount = round(self.labour_unit_rate * self.quantity, 2)
        self.total_amount = round(self.supply_amount + self.labour_amount, 2)