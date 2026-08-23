"""
FastAPI application for the ride duration prediction service.

Provides:
- POST /predict: Single prediction endpoint
- POST /predict_batch: Batch prediction endpoint
- GET /health: Health check endpoint
- Auto-generated OpenAPI documentation at /docs
"""

from fastapi import FastAPI, HTTPException, status
from datetime import datetime, timezone
import time
import uuid
import logging

from prodml.api.schemas import (
    PredictRequest,
    PredictBatchRequest,
    PredictResponse,
    PredictBatchResponse,
    HealthResponse
)
from prodml.config import settings
from prodml.predict import DurationPredictor
from prodml.logging_config import setup_logging, correlation_id_var

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Ride Duration Prediction API",
    description="Predict taxi ride duration using machine learning",
    version="0.1.0"
)

# Initialize predictor (lazy loading)
predictor = None

@app.middleware("http")
async def correlation_id_middleware(request, call_next):
    """Extracts or generates a correlation ID, stores it in contextvars, and returns header."""
    incoming_id = request.headers.get("X-Request-ID")
    correlation_id = incoming_id if incoming_id else str(uuid.uuid4())
    
    # Bind to contextvar
    token = correlation_id_var.set(correlation_id)
    
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
        
    response.headers["X-Request-ID"] = correlation_id
    return response

def get_predictor() -> DurationPredictor:
    """
    Get or initialize the predictor instance.
    
    Returns:
        DurationPredictor instance with loaded model.
    """
    global predictor
    if predictor is None:
        predictor = DurationPredictor()
        predictor.load()
    return predictor


# ---------- Health Check ----------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the service.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor._is_loaded,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ---------- Prediction Endpoints ----------
#TODO: Planning to Refactor this (to follow DRY principle)
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict duration for a single trip.
    
    Args:
        request: Trip features.
        
    Returns:
        Predicted duration.
    """
    start_time = time.perf_counter()
    
    try:
        predictor = get_predictor()
        
        # Convert Pydantic model to dict
        features = request.model_dump()
        duration = predictor.predict_one(features)

    except ValueError as e:
        logger.warning(
            "prediction_validation_error", 
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "correlation_id": correlation_id_var.get(),
                })
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except Exception as e:
        logger.error(
            "prediction_failed", 
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "correlation_id": correlation_id_var.get()
            })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal prediction error")

    processing_time_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        "prediction_success",
        extra={
            "correlation_id": correlation_id_var.get(),
            "trip_distance": request.trip_distance,
            "duration_min": duration,
            "processing_time_ms": processing_time_ms
        }
    )
    
    return PredictResponse(
        model_version=settings.model_version,
        correlation_id=correlation_id_var.get(),
        prediction=duration,
        processing_time_ms=processing_time_ms
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
async def predict_batch(request: PredictBatchRequest):
    """
    Predict duration for multiple trips.
    
    Args:
        request: List of trip features.
        
    Returns:
        List of predicted durations.
    """
    start_time = time.perf_counter()
    
    try:
        predictor = get_predictor()
        
        # Convert Pydantic models to dicts
        features_list = [trip.model_dump() for trip in request.trips]
        durations = predictor.predict_batch(features_list)
        
        
    except ValueError as e:
        logger.warning("batch_prediction_validation_error", 
                extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "correlation_id": correlation_id_var.get(),
            })
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except Exception as e:
        logger.error(
            "batch_prediction_failed", 
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "correlation_id": correlation_id_var.get()
            })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal batch prediction error")

    processing_time_ms = (time.perf_counter() - start_time) * 1000
            
    logger.info(
        "batch_prediction_success",
        extra={
            "correlation_id": correlation_id_var.get(),
            "batch_size": len(features_list),
            "processing_time_ms": processing_time_ms
        }
    )
    
    return PredictBatchResponse(
        model_version=settings.model_version,
        correlation_id=correlation_id_var.get(),
        predictions=durations,
        processing_time_ms=processing_time_ms,
        count=len(durations),
    )