"""
This module provides the DurationPredictor class for making predictions. 

The class supports:
- Single prediction: predict_one()
- Batch prediction: predict_batch()
"""

import joblib
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import logging

from prodml.config import settings
from prodml.utils import timed
from prodml.logging_config import correlation_id_var

logger = logging.getLogger(__name__)

class DurationPredictor:
    """
    This class encapsulates all prediction logic, making it easy to
    swap implementations (scikit-learn, ONNX, etc.) without changing
    the API or BentoML code.
    
    Example:
        >>> predictor = DurationPredictor()
        >>> predictor.load()
        >>> features = {'trip_distance': 5.2}
        >>> duration = predictor.predict_one(features)
        >>> print(f"Predicted duration: {duration:.2f} minutes")
        Predicted duration: 14.35 minutes
        
        >>> batch = [
        ...     {'trip_distance': 5.2},
        ...     {'trip_distance': 3.1}
        ... ]
        >>> durations = predictor.predict_batch(batch)
        >>> print(durations)
        [14.35, 8.92]
    """
    
    def __init__(self):
        """Initialize predictor with no model loaded."""
        self._model = None
        self._is_loaded = False
    
    def load(self, model_path: Optional[str] = None) -> 'DurationPredictor':
        """
        Load the trained model from disk.
        
        Returns:
            Self for method chaining.
        """
        if model_path is None:
            model_path = settings.model_path

        logger.info(
            "Loading model from disk",
            extra={
                "model_path": str(settings.model_path),
                "correlation_id": correlation_id_var.get(),
            }
                    )
        self._model = joblib.load(model_path)
        self._is_loaded = True
        logger.info(
            "Model loaded successfully",
            extra={
                "model_path": str(settings.model_path),
                "correlation_id": correlation_id_var.get(),
            }
                    )       
        return self
    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded into memory (Dry principle)."""
        if not self._is_loaded:
            logger.warning(
                "Model not loaded, loading now automatically..."و
                extra={
                    "correlation_id": correlation_id_var.get(),
                }
            )
            self.load()

    @timed
    def predict_one(self, features: Dict[str, Any]) -> float:
        """
        Predict duration for a single trip.
        
        Args:
            features: Dictionary of features.
                Example: {'trip_distance': 5.2}
        
        Returns:
            Predicted duration in minutes.
        Raises:
            ValueError: If required features are missing from any item.
        """
        self._ensure_loaded()
        # Validate required features dynamically if the model supports it, 
        # or fallback to checking essential keys
        expected_features = getattr(self._model, "feature_names_in_", ['trip_distance'])
        missing = [f for f in expected_features if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        if features.get("trip_distance", 0) > 100:
            logger.warning(
                "Input outside training range",
                extra={
                    "feature": "trip_distance",
                    "value": features.get("trip_distance", 0),
                    "training_range_max": 100,
                    "correlation_id": correlation_id_var.get(),
                }
            )
        
        values = [features[col] for col in expected_features]

        # Pass as a 2D array directly to sklearn (shape: 1, n_features)
        X = np.array([values])
        return float(self._model.predict(X)[0])
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """
        Predict duration for multiple trips.
        
        Args:
            features_list: List of feature dictionaries.
                Example: [
                    {'trip_distance': 5.2},
                    {'trip_distance': 3.1}
                ]
        
        Returns:
            List of predicted durations in minutes.
        Raises:
            ValueError: If required features are missing from any item.
        """
        self._ensure_loaded()
        
        if not features_list:
            return []
        
        # Get expected features from the model if available
        expected_features = getattr(self._model, "feature_names_in_", ['trip_distance'])
        
        # Validate and extract values in the correct order for each dict
        rows = []
        for i, features in enumerate(features_list):
            missing = [f for f in expected_features if f not in features]
            if missing:
                raise ValueError(f"Missing required features at index {i}: {missing}")
            
            rows.append([features[col] for col in expected_features])
        
        X = np.array(rows)
        return self._model.predict(X).tolist()
    
    def predict_from_df(self, df: pd.DataFrame) -> List[float]:
        """
        Predict durations from a DataFrame.
        
        Args:
            df: DataFrame with raw trip data.
        
        Returns:
            List of predicted durations.
        """
        self._ensure_loaded()
        # Get expected features from the model if available
        expected_features = getattr(self._model, "feature_names_in_", ['trip_distance'])
        
        return self._model.predict(df[expected_features]).tolist()