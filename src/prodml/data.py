"""
Data loading and splitting module.

Responsible for:
- Loading Parquet data from a file path
- Splitting data into train/validation sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
from prodml.config import settings


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load taxi trip data from a Parquet file.
    
    Args:
        file_path: Path to the Parquet file. Uses settings.data_path if not provided.
        
    Returns:
        DataFrame containing raw trip data.
        
    Examples:
        >>> df = load_data()
        >>> len(df)
        1234567
    """
    if file_path is None:
        file_path = settings.data_path
    
    df = pd.read_parquet(file_path)
    return df

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    target_col: str = 'duration_min',
    feature_cols: Optional[List[str]] = None,
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and validation sets.
    
    Args:
        df: DataFrame with features and target.
        test_size: Proportion for validation set. Uses settings if not provided.
        random_seed: Random seed for reproducibility. Uses settings if not provided.
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
        
    Examples:
        >>> X_train, X_val, y_train, y_val = split_data(df)
        >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}")
        Train: 800000, Val: 200000
    """
    if random_seed is None:
        random_seed = settings.random_seed
    if feature_cols is None:
        feature_cols = ['trip_distance']
        
    X = df[feature_cols]
    y = df[target_col]
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed
    )