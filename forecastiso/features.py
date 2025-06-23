import numpy as np
import pandas as pd
from typing import List, Dict, Union, Callable
import holidays
from abc import ABC, abstractmethod


class FeatureGenerator(ABC):
    """Base abstract class for feature generators"""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from the input dataframe"""
        pass

    @property
    def name(self) -> str:
        """Name of the feature generator"""
        return self.__class__.__name__


class LagFeatureGenerator(FeatureGenerator):
    """Generate lagged features"""

    def __init__(
        self, column: str = "load", lags: List[int] = [24, 48, 72, 168, 336, 720]
    ):
        self.column = column
        self.lags = lags

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.lags:
            df[f"{self.column}_lag_{lag}"] = df[self.column].shift(lag)
        return df


class RollingFeatureGenerator(FeatureGenerator):
    """Generate rolling window features"""

    def __init__(
        self,
        column: str = "load",
        windows: List[int] = [24, 168],
        functions: Dict[str, Callable] = None,
    ):
        self.column = column
        self.windows = windows
        self.functions = functions or {
            "mean": lambda x: x.mean(),
            "std": lambda x: x.std(),
            "min": lambda x: x.min(),
            "max": lambda x: x.max(),
        }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            for func_name, func in self.functions.items():
                df[f"{self.column}_rolling_{func_name}_{window}"] = (
                    df[self.column].rolling(window=window, min_periods=1).apply(func)
                )
        return df


class CalendarFeatureGenerator(FeatureGenerator):
    """Generate holiday-related features"""

    def __init__(self, country: str = "US"):
        self.country = country

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:

        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year
        df["dayofyear"] = df.index.dayofyear
        df["is_weekend"] = df.index.dayofweek >= 5

        # holiday stuff
        country_holidays = holidays.country_holidays(self.country)
        df["is_holiday"] = df.index.map(lambda date: date.date() in country_holidays)
        df["day_before_holiday"] = df.index.map(
            lambda date: (date.date() + pd.Timedelta(days=1)) in country_holidays
        )
        df["day_after_holiday"] = df.index.map(
            lambda date: (date.date() - pd.Timedelta(days=1)) in country_holidays
        )

        # cyclical encodings
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        return df


class InteractionFeatureGenerator(FeatureGenerator):
    """Generate interaction features between existing features"""

    def __init__(self, interactions: List[List[str]] = [["hour", "is_weekend"]]):
        self.interactions = interactions

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        for features in self.interactions:
            if len(features) == 2:
                name = f"{features[0]}_{features[1]}"
                df[name] = df[features[0]] * df[features[1]]
        return df


class FeatureManager:
    """Manager class to coordinate feature generation"""

    def __init__(self):
        self.feature_generators = []

    def add_generator(self, generator: FeatureGenerator) -> None:
        """Add a feature generator to the pipeline"""
        self.feature_generators.append(generator)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features using registered generators"""

        required_columns = {"datetime", "load"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"The dataframe must contain the following columns: {required_columns}"
            )

        # copy first so feature generators do not modify the original df
        result = df.copy()
        # check if we have a datetime index, otherwise make one
        if not isinstance(result.index, pd.DatetimeIndex):
            result.set_index("datetime", inplace=True)

        for generator in self.feature_generators:
            result = generator.generate(result)

        return result.dropna()
