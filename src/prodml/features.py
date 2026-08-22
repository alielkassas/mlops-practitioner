"""
Feature engineering module.
"""
import pandas as pd

def calc_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trip duration in minutes from pickup and dropoff timestamps.

    Args:
        df: Raw DataFrame containing 'lpep_pickup_datetime' and 
            'lpep_dropoff_datetime' columns.
        
    Returns:
        DataFrame with an added 'duration_min' column.
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'lpep_pickup_datetime': [pd.Timestamp('2026-06-01 10:00:00')],
        ...     'lpep_dropoff_datetime': [pd.Timestamp('2026-06-01 10:15:00')]
        ... })
        >>> df = calc_duration(df)
        >>> df['duration_min'].iloc[0]
        15.0
    """
    df = df.copy()
    
    # Calculate duration in minutes (target label)
    df['duration_min'] = (
        df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    ).dt.total_seconds() / 60
    
    return df