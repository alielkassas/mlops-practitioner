"""
Configuration management using pydantic-settings.

All paths, hyperparameters, and ports are loaded from environment variables
with sensible defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables are prefixed with the field name in uppercase.
    Example: DATA_PATH, MODEL_PATH, TEST_SIZE, etc.
    """
    model_config = SettingsConfigDict(
        #env_prefix="PRODML_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ---------- Data paths ----------
    data_path: Path = Field(
        default="data/green_tripdata_2023-01.parquet",
        description="Path to the Parquet data file"
    )
    
    # ---------- Model paths ----------
    model_path: Path = Field(
        default="models/model.pkl",
        description="Path to save/load the trained model"
    )

    # ---------- Logging settings ----------
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Directory to store application log files"
    )
    
    # ---------- Training parameters ----------
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    # ---------- Model type ----------
    model_type: str = Field(
        default="linear_regression",
        description="Model type: 'linear_regression' or 'random_forest'"
    )
    
    # Random Forest specific parameters
    rf_n_estimators: int = Field(
        default=100,
        description="Number of trees in Random Forest"
    )
    rf_max_depth: int = Field(
        default=10,
        description="Maximum depth of trees in Random Forest"
    )
    
    # ---------- API settings ----------
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    api_port: int = Field(
        default=8000,
        description="Port to bind the API server"
    )
        # ---------- Model type ----------
    model_version: str = Field(
        default="v0.1.0",
        description="Model version identifier"
    )
    
    


# Create a single instance of settings for the entire application
settings = Settings()