from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ClaimImageOut(BaseModel):
    id: str
    path: str
    class Config:
        orm_mode = True

class InferenceResultOut(BaseModel):
    id: str
    result: Dict[str, Any]
    cost_estimate: float
    severity: str
    class Config:
        orm_mode = True

class ClaimOut(BaseModel):
    id: str
    description: Optional[str]
    images: List[ClaimImageOut] = []
    results: List[InferenceResultOut] = []
    class Config:
        orm_mode = True
