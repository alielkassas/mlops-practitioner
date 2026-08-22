
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# ---------- Pydantic Schemas ----------
class PredictRequest(BaseModel):
    """Request body for single prediction."""
    trip_distance: float = Field(
        ...,
        gt=0,
        description="Trip distance in miles",
        example=5.2
    )
    
    model_config = ConfigDict(
        json_schema_extra = {
            "examples": [
            {
                "trip_distance": 5.2
            }
            ]
        })


class PredictBatchRequest(BaseModel):
    """Request body for batch prediction."""
    trips: List[PredictRequest] = Field(
        ...,
        description="List of trips to predict",
        min_length=1,
        max_length=100
    )
class BaseResponse(BaseModel):
    request_id: str
    processing_time_ms: float

class PredictResponse(BaseResponse):
    """Response for single prediction."""
    duration_min: float

class PredictBatchResponse(BaseResponse):
    """Response for batch prediction."""
    predictions: List[float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str