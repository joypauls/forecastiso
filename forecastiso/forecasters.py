import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from typing import Optional
from abc import ABC, abstractmethod
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings


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
    def predict(
        self, horizon: int = 24, input_features: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Predict the next 'horizon' values (default: 24 hours) with optional input features.
        Returns a pandas Series with the predictions
        """
        pass


class YesterdayForecaster(Forecaster):
    """Forecaster that uses yesterday's values as predictions"""

    def __init__(self):
        super().__init__(name="YesterdayBaseline")

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(
        self, horizon: int = 24, input_features: Optional[pd.Series] = None
    ) -> pd.Series:
        # TODO: don't hardcode column name
        return self.history.iloc[-24:]["load"].reset_index(drop=True)


class LastWeekForecaster(Forecaster):
    """Forecaster that uses last week's values as predictions"""

    def __init__(self):
        super().__init__(name="LastWeekBaseline")

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(
        self, horizon: int = 24, input_features: Optional[pd.Series] = None
    ) -> pd.Series:
        # TODO: don't hardcode column name
        return self.history.iloc[-24 * 7 : -24 * 6]["load"].reset_index(drop=True)


class RollingMeanForecaster(Forecaster):
    """Forecaster that uses the rolling mean of past days at the same hour"""

    def __init__(self, window_days=4):
        super().__init__(name=f"RollingMean{window_days}dBaseline")
        self.window_days = window_days

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(
        self, horizon: int = 24, input_features: Optional[pd.Series] = None
    ) -> pd.Series:
        df = self.history.copy()
        forecasts = []

        for h in range(24):
            # TODO: don't hardcode column name
            hourly_vals = df[df.index.hour == h]["load"][-self.window_days * 24 :]
            forecasts.append(hourly_vals.mean())

        return pd.Series(forecasts[:horizon])


class ARIMAForecaster(Forecaster):
    """
    ARIMA forecaster for electrical load data with configurable parameters.
    Uses SARIMA(p,d,q)(P,D,Q,s) for seasonal patterns.
    (p,d,q) corresponds to 'order' and (P,D,Q,s) to 'seasonal_order'.
    """

    def __init__(
        self,
        target_col: str = "load",
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (0, 1, 1, 24),
        min_history_hours: int = 24 * 7,
    ):
        super().__init__(name="ARIMABaseline")
        self.target_col = target_col
        self.order = order
        self.seasonal_order = seasonal_order
        self.min_history_hours = min_history_hours
        self.model = None

    def fit(self, history: pd.DataFrame):
        """Fit ARIMA model to historical load data"""
        self.history = history.copy()

        if self.target_col not in history.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        if len(history) < self.min_history_hours:
            raise ValueError(f"Need at least {self.min_history_hours} hours of data")

        target_values = history[self.target_col].dropna().values

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            arima_model = ARIMA(
                target_values,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model = arima_model.fit(method_kwargs={"warn_convergence": False})

    def predict(
        self, horizon: int = 24, input_features: Optional[pd.Series] = None
    ) -> pd.Series:
        """Predict next 'horizon' hours using fitted ARIMA model"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.model.forecast(steps=horizon)
        forecast = np.maximum(forecast, 0)

        return pd.Series(forecast, name=self.target_col)


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
        super().__init__(name="XGBoostBaseline")
        self.target_col = target_col
        self.feature_cols = feature_cols or []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.train_interval = 24
        self.model = None

    def fit(self, history: pd.DataFrame):
        self.history = history.copy().reset_index(drop=True)

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
            features = self.history[self.feature_cols].iloc[i - 1].values

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
        if input_features is None:
            raise ValueError("Input features must be provided for prediction.")

        missing_cols = [
            col for col in self.feature_cols if col not in input_features.index
        ]
        if missing_cols:
            raise ValueError(f"Feature columns not found in input: {missing_cols}")

        features = input_features[self.feature_cols].values

        X_input = features.reshape(1, -1)
        forecast = self.model.predict(X_input)[0]
        forecast = np.maximum(forecast, 0)

        return pd.Series(forecast[:horizon])
