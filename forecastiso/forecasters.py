import numpy as np
import pandas as pd

# from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from xgboost import XGBRegressor


class Forecaster(ABC):
    """Base abstract class for all forecasters"""

    def __init__(self, name="UnnamedForecaster"):
        self.name = name
        self.history = None

    @abstractmethod
    def fit(self, history: pd.DataFrame):
        """Fit the forecaster to historical data"""
        pass

    @abstractmethod
    def predict(self, horizon: int = 24) -> pd.Series:
        """
        Predict the next 'horizon' values (default: 24 hours)
        Returns a pandas Series with the predictions
        """
        pass


class NaiveYesterdayForecaster(Forecaster):
    """Forecaster that uses yesterday's values as predictions"""

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(self, horizon: int = 24) -> pd.Series:
        # TODO: don't hardcode column name
        return self.history.iloc[-24:]["load"].reset_index(drop=True)


class NaiveLastWeekForecaster(Forecaster):
    """Forecaster that uses last week's values as predictions"""

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(self, horizon: int = 24) -> pd.Series:
        # TODO: don't hardcode column name
        return self.history.iloc[-24 * 7 : -24 * 6]["load"].reset_index(drop=True)


class RollingMeanForecaster(Forecaster):
    """Forecaster that uses the rolling mean of past days at the same hour"""

    def __init__(self, window_days=4):
        super().__init__(name=f"RollingMean_{window_days}d")
        self.window_days = window_days

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(self, horizon: int = 24) -> pd.Series:
        df = self.history.copy()
        forecasts = []

        for h in range(24):
            # TODO: don't hardcode column name
            hourly_vals = df[df.index.hour == h]["load"][-self.window_days * 24 :]
            forecasts.append(hourly_vals.mean())

        return pd.Series(forecasts[:horizon])


# class SimpleXGBForecaster(Forecaster):
#     """
#     Forecasts the next 24 hours of load using features at hour 23 of each day.
#     Trained with one sample per day: X at hour 23 → y for hours 0–23 of the next day.
#     """

#     def __init__(
#         self,
#         feature_cols: Optional[list[str]] = None,
#         target_col: str = "load",
#         n_estimators: int = 200,
#         learning_rate: float = 0.05,
#         max_depth: int = 5,
#         standardize: bool = True,
#     ):
#         super().__init__(name="XGBoost")

#         self.feature_cols = feature_cols
#         self.target_col = target_col
#         self.n_estimators = n_estimators
#         self.learning_rate = learning_rate
#         self.max_depth = max_depth
#         self.standardize = standardize
#         self.model = None
#         self.scaler = None
#         self.bool_cols = []
#         self.numeric_cols = []

#     def fit(self, history: pd.DataFrame):
#         """Train model on features at hour 23 to predict next day's 24-hour load."""
#         self.history = history.copy().reset_index(drop=True)

#         if self.history.empty:
#             raise ValueError("History is empty. Cannot fit model.")
#         if self.target_col not in history.columns:
#             raise ValueError(f"Target column '{self.target_col}' not found in history.")

#         if self.feature_cols is None:
#             self.feature_cols = [
#                 col
#                 for col in history.columns
#                 if col != self.target_col and col != "area"
#             ]
#         else:
#             missing_cols = [
#                 col for col in self.feature_cols if col not in history.columns
#             ]
#             if missing_cols:
#                 raise ValueError(
#                     f"Feature columns {missing_cols} not found in history."
#                 )

#         self.bool_cols = [
#             col for col in self.feature_cols if history[col].dtype == "bool"
#         ]
#         self.numeric_cols = [
#             col for col in self.feature_cols if col not in self.bool_cols
#         ]

#         scaled_history = self.history[self.feature_cols + [self.target_col]].copy()

#         if self.standardize:
#             if self.bool_cols:
#                 scaled_history[self.bool_cols] = scaled_history[self.bool_cols].astype(
#                     int
#                 )
#             if self.numeric_cols:
#                 self.scaler = StandardScaler()
#                 scaled_history[self.numeric_cols] = self.scaler.fit_transform(
#                     scaled_history[self.numeric_cols]
#                 )

#         # Train using hour 23 of each day
#         X_train, y_train = [], []
#         for i in range(
#             23, len(scaled_history) - 24, 24
#         ):  # every day, starting from hour 23
#             features = scaled_history.iloc[i][self.feature_cols].values
#             next_24h = scaled_history.iloc[i + 1 : i + 25][self.target_col].values

#             if len(next_24h) == 24:
#                 X_train.append(features)
#                 y_train.append(next_24h)

#         self.model = MultiOutputRegressor(
#             XGBRegressor(
#                 n_estimators=self.n_estimators,
#                 learning_rate=self.learning_rate,
#                 max_depth=self.max_depth,
#                 objective="reg:squarederror",
#                 verbosity=0,
#                 booster="gbtree",
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 reg_lambda=1.0,
#                 reg_alpha=0.0,
#                 random_state=42,
#             )
#         )
#         self.model.fit(np.array(X_train), np.array(y_train))

#     def predict(
#         self, horizon: int = 24, external_features: Optional[pd.Series] = None
#     ) -> pd.Series:
#         """
#         Predict next day's 24-hour load using either the latest hour-23 features or external features.

#         Args:
#             horizon: Number of hours to predict (must be <= 24)
#             external_features: Optional external feature values to use instead of latest history

#         Returns:
#             Pandas Series with predicted values
#         """
#         if self.model is None:
#             raise ValueError("Model not fitted. Call fit() first.")
#         if horizon > 24:
#             raise ValueError("Horizon must be 24 or less.")

#         # Use external features if provided, otherwise use latest from history
#         if external_features is not None:
#             # Verify all required features are present
#             missing_cols = [
#                 col for col in self.feature_cols if col not in external_features
#             ]
#             if missing_cols:
#                 raise ValueError(f"Missing required features: {missing_cols}")

#             input_features = pd.DataFrame([external_features])[self.feature_cols]
#         else:
#             latest = self.history.iloc[-1]  # Assumes this is hour 23 of the current day
#             input_features = pd.DataFrame([latest])[self.feature_cols]

#         # Apply the same preprocessing as during training
#         if self.standardize:
#             if self.bool_cols:
#                 input_features[self.bool_cols] = input_features[self.bool_cols].astype(
#                     int
#                 )
#             if self.numeric_cols:
#                 input_features[self.numeric_cols] = self.scaler.transform(
#                     input_features[self.numeric_cols]
#                 )

#         features = input_features.values.reshape(1, -1)
#         predictions = self.model.predict(features)[0]

#         return pd.Series(predictions[:horizon])


class XGBForecaster(Forecaster):
    """
    Uses any provided features to predict the next 24 hours using XGBoost.
    Can work with window features generated by WindowFeatureGenerator or any other features.
    """

    def __init__(
        self,
        target_col: str = "load",
        feature_cols: Optional[list[str]] = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 5,
    ):
        super().__init__(name="WindowedXGBoost")
        self.target_col = target_col
        self.feature_cols = feature_cols or []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.train_interval = 24
        self.model = None

    def fit(self, history: pd.DataFrame):
        self.history = history.copy().reset_index(drop=True)

        # Validate feature columns
        missing_cols = [
            col for col in self.feature_cols if col not in self.history.columns
        ]
        if missing_cols:
            raise ValueError(f"Feature columns not found in data: {missing_cols}")

        X_train, y_train = [], []
        for i in range(
            self.train_interval,
            len(self.history) - self.train_interval,
            self.train_interval,
        ):
            # Get feature values at position i-1 (right before we want to predict)
            features = self.history[self.feature_cols].iloc[i - 1].values

            # Get future values to predict
            future_values = (
                self.history[self.target_col].iloc[i : i + self.train_interval].values
            )

            X_train.append(features)
            y_train.append(future_values)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        base_model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            objective="reg:squarederror",
            verbosity=0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=1729,
        )

        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train, y_train)

    def predict(
        self, horizon: int = 24, input_features: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Predict next 24 hours using the provided features.

        Args:
            horizon: Number of hours to predict (must be <= train_interval)
            input_features: Optional Series with feature values. Subsets to feature_cols.

        Returns:
            Pandas Series with predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if horizon > self.train_interval:
            raise ValueError(f"Horizon must be {self.train_interval} or less.")

        # Verify all required features are present
        missing_cols = [
            col for col in self.feature_cols if col not in input_features.index
        ]
        if missing_cols:
            raise ValueError(f"Feature columns not found in input: {missing_cols}")

        # Extract feature values
        features = input_features[self.feature_cols].values

        # Reshape for prediction
        X_input = features.reshape(1, -1)
        prediction = self.model.predict(X_input)[0]

        return pd.Series(prediction[:horizon])
