"""
Pydantic schemas for the API.

Defines request and response models with validation and OpenAPI documentation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# ---------- Pydantic Schemas ----------
class PredictRequest(BaseModel):
    """Request body for single prediction."""
    trip_distance: float = Field(
        ...,
        gt=0,
        lt=100,
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
    model_config = {
        "json_schema_extra": {
            "example": {
                "trips": [
                    {"trip_distance": 5.2},
                    {"trip_distance": 3.1}
                ]
            }
        }
    }

# ---------- Response Schemas ----------
class BaseResponse(BaseModel):
    model_version: str = Field(
        ...,
        description="Model version identifier",
        example="v0.1.0"
    )
    
    correlation_id: str = Field(
        ...,
        description="Unique request ID for tracking",
        example="a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds",
        example=45.2
    )


class PredictResponse(BaseResponse):
    """
    Single prediction response.
    """
    
    prediction: float = Field(
        ...,
        description="Predicted duration in minutes",
        example=14.35
    )

class PredictBatchResponse(BaseResponse):
    """
    Batch prediction response.
    """
    
    predictions: List[float] = Field(
        ...,
        description="List of predicted durations",
        example=[14.35, 25.12]
    )
    
    count: int = Field(
        ...,
        description="Number of predictions",
        example=2
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "trips": [
                    {"trip_distance": 5.2},
                    {"trip_distance": 3.1}
                ]
            }
        }
    }


# ---------- Health Schemas ----------
class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded in memory",
        example=True
    )
    
    timestamp: str = Field(
        ...,
        description="Current UTC timestamp"
    )


class MetadataResponse(BaseModel):
    """Model metadata response."""
    
    model_version: str = Field(
        ...,
        description="Model version",
        example="v0.1.0"
    )
    
    training_date: str = Field(
        ...,
        description="Model training date",
        example="2024-01-15T10:30:00.123Z"
    )
    
    features: List[str] = Field(
        ...,
        description="Feature names used by the model",
        example=["trip_distance"]
    )
    
    framework: str = Field(
        ...,
        description="ML framework used",
        example="scikit-learn"
    )
    
    artifact_hash: str = Field(
        ...,
        description="Model file hash for versioning",
        example="a1b2c3d4e5f6"
    )