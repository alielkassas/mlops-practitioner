"""
FastAPI application for the ride duration prediction service.

Provides:
- POST /predict: Single prediction endpoint
- POST /predict_batch: Batch prediction endpoint
- GET /health: Health check endpoint
- GET /metadata: Model metadata endpoint
- Auto-generated OpenAPI documentation at /docs
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import time
import uuid
import logging
import os
import hashlib
from contextlib import asynccontextmanager

from prodml.api.schemas import (
    PredictRequest,
    PredictBatchRequest,
    PredictResponse,
    PredictBatchResponse,
    HealthResponse,
    MetadataResponse,
)
from prodml.config import settings
from prodml.predict import DurationPredictor
from prodml.logging_config import setup_logging, correlation_id_var

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# ---------- Global Predictor ----------
predictor: DurationPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading the model once at startup.
    
    This is the correct way to load models - once at startup, not per request.
    Loading per-request is the most common beginner mistake.
    """
    global predictor
    
    logger.info("Starting up API server and loading ML model into memory...")
    try:
        predictor = DurationPredictor()
        predictor.load()
        logger.info("Model loaded successfully into memory.")
    except Exception as e:
        logger.error("Failed to load model at startup", extra={"error": str(e)})
    
    yield  # Application runs here
    
    # Cleanup if needed
    logger.info("Shutting down API server...")


app = FastAPI(
    title="Ride Duration Prediction API",
    description="Predict taxi ride duration using machine learning",
    version="0.1.0",
    lifespan=lifespan,
)

# Initialize predictor (lazy loading)
predictor = None

@app.middleware("http")
async def correlation_id_middleware(request:Request, call_next):
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

    logger.info(
            "Request completed",
            extra={
                "correlation_id": correlation_id,
                "status_code": response.status_code
            }
        )
    
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors and return a structured 422 response.

    Args:
        request (Request): The incoming FastAPI HTTP request object.
        exc (RequestValidationError): The validation exception containing details 
            about the failed validation rules.

    Returns:
        JSONResponse: A FastAPI JSON response with status code 422 containing 
        a descriptive message and a list of validation error items.
    
    Example:
        If a user sends a negative trip distance, this handler formats the output as:
        {
            "message": "Validation error in request parameters",
            "errors": [...]
        }
    """
    logger.warning("validation_error", 
                   extra={
                       "error": str(exc), 
                        "path": request.url.path,
                        "method": request.method,
                        "client_ip": request.client.host if request.client else "unknown",
                       "correlation_id": correlation_id_var.get(),
                       }
                    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Validation error in request parameters",
            "errors": exc.errors(),
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions and return a structured 500 response.

    Args:
        request (Request): The incoming FastAPI HTTP request object.
        exc (Exception): The unexpected exception that occurred during request processing.
    Returns:
        JSONResponse: A FastAPI JSON response with status code 500 containing
        a generic error message and the current request correlation ID.
    
    Example:
        If an unexpected database or model error occurs, this handler formats 
        the client response as:
        {
            "message": "Internal server error.",
            "correlation_id": "13d27f8f-1058-4b77-856e-d55fae76eb42"
        }
    """
    logger.error("internal_server_error", 
                 exc_info=True, 
                 extra={
                     "error": str(exc), 
                     "path": request.url.path,
                     "method": request.method,
                     "correlation_id": correlation_id_var.get(),
                }
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error.",
            "correlation_id": correlation_id_var.get(),
        }
    )
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

# ---------- Metadata ----------
@app.get("/metadata", response_model=MetadataResponse)
async def metadata():
    """
    Get model metadata.
    
    Returns:
        - model_version: Current version
        - training_date: When the model was trained
        - features: Feature names
        - framework: ML framework used
        - artifact_hash: Hash of the model file
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    # Calculate file hash
    model_hash = "unknown"
    if os.path.exists(settings.model_path):
        with open(settings.model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    
    return MetadataResponse(
        model_version=settings.model_version or "v0.1.0",
        training_date=datetime.utcnow().isoformat() + "Z",
        features=["trip_distance"],
        framework="scikit-learn",
        artifact_hash=model_hash
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