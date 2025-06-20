import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
import holidays


def generate_features(
    df: pd.DataFrame,
    lags: List[int] = [24, 48, 72, 168, 336, 720],
    rolling_windows: List[int] = [24, 168],
    add_holidays: bool = True,
    country: str = "US",
) -> pd.DataFrame:
    """
    Generate comprehensive temporal features for ISO load forecasting.

    Args:
        df: DataFrame with 'load' column and datetime index
        lags: List of hourly lag values to include
        rolling_windows: List of window sizes for rolling statistics
        add_holidays: Whether to add holiday features
        country: Country code for holidays
        add_cyclical: Whether to encode cyclical features using sin/cos transformation

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # Add lagged values
    for lag in lags:
        df[f"lag_{lag}"] = df["load"].shift(lag)

    # Add rolling statistics
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = (
            df["load"].rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = (
            df["load"].rolling(window=window, min_periods=1).std()
        )
        df[f"rolling_max_{window}"] = (
            df["load"].rolling(window=window, min_periods=1).max()
        )
        df[f"rolling_min_{window}"] = (
            df["load"].rolling(window=window, min_periods=1).min()
        )

    # Calendar features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = df.index.dayofweek >= 5

    # Add holiday information
    if add_holidays:
        try:
            country_holidays = holidays.country_holidays(country)
            df["is_holiday"] = df.index.date.map(lambda date: date in country_holidays)

            # Get days before and after holidays
            df["day_before_holiday"] = df.index.date.map(
                lambda date: (date + pd.Timedelta(days=1)) in country_holidays
            )
            df["day_after_holiday"] = df.index.date.map(
                lambda date: (date - pd.Timedelta(days=1)) in country_holidays
            )
        except Exception:
            # Fallback if holidays package not available
            df["is_holiday"] = False
            df["day_before_holiday"] = False
            df["day_after_holiday"] = False

    # Handle special days (could be extended)
    df["is_first_or_last_day"] = (df.index.day == 1) | (
        df.index.day == pd.Index(df.index).to_series().dt.days_in_month
    )

    # Add interaction features
    df["hour_weekday"] = df["hour"] * (df.index.dayofweek < 5)
    df["hour_weekend"] = df["hour"] * (df.index.dayofweek >= 5)

    return df.dropna()
