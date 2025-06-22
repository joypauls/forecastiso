import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
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


class LinearRegressionForecaster(Forecaster):
    """
    Forecaster that uses a single linear regression model to predict all 24 hours at once.
    Currently only supports numeric and boolean features.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "load",
        use_ridge: bool = False,
        alpha: float = 1.0,
        standardize: bool = True,
    ):
        name = "Ridge" if use_ridge else "Linear"
        name += "Regression"
        super().__init__(name=name)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.standardize = standardize
        self.model = None
        self.scaler = None
        self.bool_cols = []
        self.numeric_cols = []

    def fit(self, history: pd.DataFrame):
        """Fit a regression model that predicts 24 hours at once"""
        self.history = history.copy()

        if self.history.empty:
            raise ValueError("History is empty. Cannot fit model.")
        if self.target_col not in history.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in history.")
        if self.feature_cols is not None:
            missing_cols = [
                col for col in self.feature_cols if col not in history.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Feature columns {missing_cols} not found in history."
                )

        if self.feature_cols is None:
            self.feature_cols = [
                col
                for col in history.columns
                if col != self.target_col and col != "area"
            ]

        self.bool_cols = [
            col for col in self.feature_cols if history[col].dtype == "bool"
        ]
        self.numeric_cols = [
            col for col in self.feature_cols if col not in self.bool_cols
        ]

        print(self.bool_cols)
        print(self.numeric_cols)

        self.history = self.history.reset_index(drop=True)
        self.history = self.history[self.feature_cols + [self.target_col]]

        scaled_history = self.history.copy()

        if self.standardize:
            if self.bool_cols:
                scaled_history[self.bool_cols] = scaled_history[self.bool_cols].astype(
                    int
                )
            if self.numeric_cols:
                self.scaler = StandardScaler()
                scaled_history[self.numeric_cols] = self.scaler.fit_transform(
                    scaled_history[self.numeric_cols]
                )

        step = 24
        X_train, y_train = [], []
        for i in range(0, len(scaled_history) - 24, step):
            features = scaled_history.iloc[i][self.feature_cols].values
            next_24h = scaled_history.iloc[i + 1 : i + 25][self.target_col].values

            if len(next_24h) == 24:
                X_train.append(features)
                y_train.append(next_24h)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        base_model = Ridge(alpha=self.alpha) if self.use_ridge else LinearRegression()
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train, y_train)

    def predict(self, horizon: int = 24) -> pd.Series:
        """Predict next 24 hours of load"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if horizon > 24:
            raise ValueError("Horizon must be 24 or less.")

        latest = self.history.iloc[-1 : self.history.shape[0]].copy()

        if self.standardize:
            if self.bool_cols:
                latest[self.bool_cols] = latest[self.bool_cols].astype(int)
            if self.numeric_cols:
                latest[self.numeric_cols] = self.scaler.transform(
                    latest[self.numeric_cols]
                )

        features = latest[self.feature_cols].values.reshape(1, -1)
        predictions = self.model.predict(features)[0]

        return pd.Series(predictions[:horizon])


class GradientBoostingForecaster(Forecaster):
    """
    Forecaster that uses XGBoost gradient boosting to predict all 24 hours at once.
    Should outperform linear regression by capturing non-linear patterns.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "load",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        standardize: bool = True,
    ):
        super().__init__(name="XGBoost")

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.standardize = standardize
        self.model = None
        self.scaler = None
        self.bool_cols = []
        self.numeric_cols = []

    def fit(self, history: pd.DataFrame):
        """Fit an XGBoost model that predicts 24 hours at once"""
        self.history = history.copy()

        # Validation checks
        if self.history.empty:
            raise ValueError("History is empty. Cannot fit model.")
        if self.target_col not in history.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in history.")

        # Determine features to use
        if self.feature_cols is None:
            self.feature_cols = [
                col
                for col in history.columns
                if col != self.target_col and col != "area"
            ]
        else:
            missing_cols = [
                col for col in self.feature_cols if col not in history.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Feature columns {missing_cols} not found in history."
                )

        # Identify boolean and numeric columns
        self.bool_cols = [
            col for col in self.feature_cols if history[col].dtype == "bool"
        ]
        self.numeric_cols = [
            col for col in self.feature_cols if col not in self.bool_cols
        ]

        # Prepare dataset
        self.history = self.history.reset_index(drop=True)
        self.history = self.history[self.feature_cols + [self.target_col]]
        scaled_history = self.history.copy()

        # Standardize if needed
        if self.standardize:
            # Convert booleans to integers
            if self.bool_cols:
                scaled_history[self.bool_cols] = scaled_history[self.bool_cols].astype(
                    int
                )

            # Scale numeric features
            if self.numeric_cols:
                self.scaler = StandardScaler()
                scaled_history[self.numeric_cols] = self.scaler.fit_transform(
                    scaled_history[self.numeric_cols]
                )

        # Create training data
        step = 24  # Hours in a day
        X_train, y_train = [], []
        for i in range(0, len(scaled_history) - 24, step):
            features = scaled_history.iloc[i][self.feature_cols].values
            next_24h = scaled_history.iloc[i + 1 : i + 25][self.target_col].values

            if len(next_24h) == 24:
                X_train.append(features)
                y_train.append(next_24h)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Configure XGBoost for best performance with time series
        # Remove early_stopping_rounds parameter which requires validation data
        base_model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            objective="reg:squarederror",
            verbosity=0,
            booster="gbtree",
            subsample=0.8,
            colsample_bytree=0.8,
            # Regularization to prevent overfitting
            reg_lambda=1.0,
            reg_alpha=0.0,
            # No early stopping since we don't have validation data
            # For reproducibility
            random_state=42,
        )

        # Create a multi-output model to predict all 24 hours at once
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train, y_train)

    def predict(self, horizon: int = 24) -> pd.Series:
        """Predict next 24 hours of load"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if horizon > 24:
            raise ValueError("Horizon must be 24 or less.")

        # Get the most recent data point
        latest = self.history.iloc[-1 : self.history.shape[0]].copy()

        # Apply same preprocessing as during training
        if self.standardize:
            if self.bool_cols:
                latest[self.bool_cols] = latest[self.bool_cols].astype(int)
            if self.numeric_cols:
                latest[self.numeric_cols] = self.scaler.transform(
                    latest[self.numeric_cols]
                )

        # Make the prediction
        features = latest[self.feature_cols].values.reshape(1, -1)
        predictions = self.model.predict(features)[0]

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
