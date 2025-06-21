import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Optional
from abc import ABC, abstractmethod


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
        return self.history.iloc[-24:]["load"].reset_index(drop=True)


class NaiveLastWeekForecaster(Forecaster):
    """Forecaster that uses last week's values as predictions"""

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(self, horizon: int = 24) -> pd.Series:
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
            hourly_vals = df[df.index.hour == h]["load"][-self.window_days * 24 :]
            forecasts.append(hourly_vals.mean())

        return pd.Series(forecasts[:horizon])


class LinearRegressionForecaster(Forecaster):
    """Forecaster that uses a single linear regression model to predict all 24 hours at once"""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        use_ridge: bool = False,
        alpha: float = 1.0,
        standardize: bool = True,
    ):
        name = "Ridge" if use_ridge else "Linear"
        name += "Regression"
        if standardize:
            name += "_std"
        super().__init__(name=name)

        self.feature_cols = feature_cols
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.standardize = standardize
        self.model = None

    def fit(self, history: pd.DataFrame):
        """Fit a regression model that predicts 24 hours at once"""
        self.history = history.copy()

        if self.feature_cols is None:
            self.feature_cols = [
                col for col in history.columns if col != "load" and col != "area"
            ]

        X_train = []
        y_train = []

        # for each day in history, create a training sample
        # that maps features at time t to next 24 hours of load at t+1...t+24
        for i in range(len(history) - 24):
            features = history.iloc[i][self.feature_cols].values
            next_24h = history.iloc[i + 1 : i + 25]["load"].values

            if len(next_24h) < 24:
                continue

            X_train.append(features)
            y_train.append(next_24h)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if self.use_ridge:
            base_model = Ridge(alpha=self.alpha)
        else:
            base_model = LinearRegression()

        # predict all 24 hours at once
        model = MultiOutputRegressor(base_model)

        if self.standardize:
            self.model = Pipeline([("scaler", StandardScaler()), ("regressor", model)])
        else:
            self.model = model

        self.model.fit(X_train, y_train)

    def predict(self, horizon: int = 24) -> pd.Series:
        """Predict next 24 hours of load"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        latest_data = self.history.iloc[-1][self.feature_cols].values.reshape(1, -1)

        predictions = self.model.predict(latest_data)[0]

        return pd.Series(predictions[:horizon])


# class EnsembleForecaster(Forecaster):
#     """Combines multiple forecasters using weighted average"""

#     def __init__(
#         self, forecasters: List[Forecaster], weights: Optional[List[float]] = None
#     ):
#         super().__init__(name="Ensemble")
#         self.forecasters = forecasters

#         if weights is None:
#             self.weights = [1.0 / len(forecasters)] * len(forecasters)
#         else:
#             total = sum(weights)
#             self.weights = [w / total for w in weights]

#     def fit(self, history: pd.DataFrame):
#         """Fit all component forecasters"""
#         self.history = history

#         for forecaster in self.forecasters:
#             forecaster.fit(history)

#     def predict(self, horizon: int = 24) -> pd.Series:
#         """Predict using weighted average of component forecasters"""
#         predictions = []

#         for forecaster in self.forecasters:
#             pred = forecaster.predict(horizon)
#             predictions.append(pred)

#         ensemble_pred = sum(p * w for p, w in zip(predictions, self.weights))

#         return ensemble_pred
