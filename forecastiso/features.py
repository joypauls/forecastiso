import pandas as pd


def generate_lagged_calendar_features(df: pd.DataFrame, lags=[24, 168]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["load"].shift(lag)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    return df.dropna()
