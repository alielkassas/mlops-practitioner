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

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import calc_duration
from prodml.utils import timed
import logging
from prodml.logging_config import setup_logging

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
    # 1. Load and clean data
    df = load_data()
    df = calc_duration(df)
    logger.info(f"Loaded {len(df)} rows")
    
    # 2. Split data
    X_train, X_val, y_train, y_val = split_data(df)
    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
    
    # 5. Train model
    model = get_model()
    model.fit(X_train, y_train)
    logger.info(f"Model training completed: {type(model).__name__}")
    
    # 6. Evaluate
    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    logger.info(f"Validation RMSE: {rmse:.2f}")
    logger.info(f"Validation MAE: {mae:.2f}")
    
    # 7. Save model
    os.makedirs(settings.model_path.parent, exist_ok=True)
    joblib.dump(model, settings.model_path)
    
    logger.info(f"Model saved to {settings.model_path}")
    
    return rmse, mae


def main() -> None:
    """Entry point for the prodml-train command."""
    rmse, mae = train_model()
    print(f"   RMSE: {rmse:.2f} minutes")
    print(f"   MAE:  {mae:.2f} minutes")

if __name__ == "__main__":
    main()