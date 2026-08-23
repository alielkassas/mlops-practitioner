"""
Model training and persistence module.

Responsible for:
- Training the model with the chosen algorithm
- Saving the trained model to disk
- Computing and reporting validation metrics
"""

import joblib
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import Tuple, Any
import logging


from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import calc_duration
from prodml.utils import timed
from prodml.logging_config import setup_logging, correlation_id_var

setup_logging()
logger = logging.getLogger(__name__)

def get_model() -> Any:
    """
    Instantiate the model based on settings.
    
    Returns:
        The model instance (LinearRegression or RandomForestRegressor).
        
    Raises:
        ValueError: If model_type is not supported.
    """
    if settings.model_type == "linear_regression":
        return LinearRegression()
    elif settings.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=settings.rf_n_estimators,
            max_depth=settings.rf_max_depth,
            random_state=settings.random_seed
        )
    else:
        raise ValueError(f"Unknown model_type: {settings.model_type}")


@timed
def train_model() -> Tuple[float, float]:
    """
    Train the model and save it to disk.
    
    Returns:
        Tuple of (rmse, mae) on validation set.
    """
    # Load and clean data
    df = load_data()
    df = calc_duration(df)

    logger.info(
        "Data loaded successfully",
        extra={
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "correlation_id": correlation_id_var.get(),
        }
        )
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(df)
    logger.info(
        "Data splitted successfully",
        extra={
            "train_size": len(X_train),
            "val_size": len(X_val),
            "correlation_id": correlation_id_var.get(),
        }
    )
    
    # Train model
    model = get_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    logger.info(
        "Model training completed successfully",
        extra={
            "model_type": type(model).__name__,
            "rmse": rmse,
            "mae": mae,
            "correlation_id": correlation_id_var.get(),
        }
    )
    
    # Save model
    os.makedirs(settings.model_path.parent, exist_ok=True)
    joblib.dump(model, settings.model_path)
    
    logger.info(
        "Model saved successfully", 
        extra={
            "path": str(settings.model_path),
            "correlation_id": correlation_id_var.get(),
        }
    )
    
    return rmse, mae


def main() -> None:
    """Entry point for the prodml-train command."""
    rmse, mae = train_model()

if __name__ == "__main__":
    main()