from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

@dataclass
class BOQItemWithEmbedding:
    item_id: int
    boq_id: int
    item_code: Optional[str]
    item_description: str
    unit_of_measurement: str
    quantity: Decimal
    supply_unit_rate: Optional[Decimal]
    supply_amount: Optional[Decimal]
    labour_unit_rate: Optional[Decimal]
    labour_amount: Optional[Decimal]
    total_amount: Optional[Decimal]
    location_id: Optional[int]
    created_by: Optional[str]
    created_at: Optional[datetime]
    updated_by: Optional[str]
    updated_at: Optional[datetime]
    description_embedding: Optional[List[float]] = None
    embedding_generated_at: Optional[datetime] = None