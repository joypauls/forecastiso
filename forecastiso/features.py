import pandas as pd
from typing import List, Dict, Callable
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
    """
    Generate rolling window features. Allows custom functions to apply to the windows.
    """

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


class WindowFeatureGenerator(FeatureGenerator):
    """Generate features containing windows of past values."""

    def __init__(self, column: str = "load", window_sizes: List[int] = [24]):
        self.column = column
        self.window_sizes = window_sizes

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for size in self.window_sizes:
            for i in range(size):
                result[f"{self.column}_window_{size}_{i}"] = df[self.column].shift(
                    size - i
                )

        return result


class CalendarFeatureGenerator:
    """
    Generate calendar and holiday-related features, including forward-shifted target versions.
    Right now this has no special handling for daylight saving time (DST).
    """

    def __init__(self, country: str = "US"):
        self.country = country

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # base calendar features
        df["hour"] = df.index.hour
        df["dow"] = df.index.dayofweek
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year
        df["doy"] = df.index.dayofyear
        df["is_weekend"] = df["dow"] >= 5
        df["day_before_weekend"] = df["dow"].isin([4, 5])

        # holiday features
        country_holidays = holidays.country_holidays(self.country)
        df["is_holiday"] = df.index.map(lambda date: date.date() in country_holidays)
        df["day_before_holiday"] = df.index.map(
            lambda date: (date.date() + pd.Timedelta(days=1)) in country_holidays
        )
        df["day_after_holiday"] = df.index.map(
            lambda date: (date.date() - pd.Timedelta(days=1)) in country_holidays
        )

        shift_hours = -24
        calendar_cols = [
            "dow",
            "month",
            "day",
            "quarter",
            "year",
            "doy",
            "is_weekend",
            "day_before_weekend",
            "is_holiday",
            "day_before_holiday",
            "day_after_holiday",
        ]
        for col in calendar_cols:
            # this fails in the spring daylight saving time change
            df[f"target_{col}"] = df[col].shift(shift_hours)

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
        # TODO: should this be done in the data loader?
        # TODO: handle timezones and DST correctly
        if not isinstance(result.index, pd.DatetimeIndex):
            result.set_index("datetime", inplace=True)

        for generator in self.feature_generators:
            result = generator.generate(result)

        return result.dropna()
