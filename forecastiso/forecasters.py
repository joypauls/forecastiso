import numpy as np
import pandas as pd
from forecastiso.forecasters import Forecaster
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from forecastiso.features import generate_lagged_calendar_features


class NaiveYesterdayForecaster(Forecaster):
    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(self, horizon: int) -> pd.Series:
        return self.history.iloc[-24:]["load"].reset_index(drop=True)


class NaiveLastWeekForecaster(Forecaster):
    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(self, horizon: int) -> pd.Series:
        return self.history.iloc[-24 * 7 : -24 * 6]["load"].reset_index(drop=True)


class RollingMeanForecaster(Forecaster):
    def __init__(self, window_days=4):
        super().__init__(name=f"RollingMean_{window_days}d")
        self.window_days = window_days

    def fit(self, history: pd.DataFrame):
        self.history = history

    def predict(self, horizon: int) -> pd.Series:
        df = self.history.copy()
        forecasts = []
        for h in range(24):
            hourly_vals = df[df.index.hour == h]["load"][-self.window_days * 24 :]
            forecasts.append(hourly_vals.mean())
        return pd.Series(forecasts)


class LinearRegressionForecaster(Forecaster):
    def __init__(self, name="LinearRegression", lags=[24, 168]):
        super().__init__(name)
        self.lags = lags
        self.model = None

    def fit(self, history: pd.DataFrame):
        self.history = history
        df = generate_lagged_calendar_features(history, self.lags)
        X = df.drop(columns=["load"])
        y = df["load"].values

        X_list = []
        y_list = []
        for i in range(len(X) - 23):
            X_list.append(X.iloc[i])
            y_list.append(history["load"].iloc[i + 1 : i + 25].values)

        X_final = pd.DataFrame(X_list)
        y_final = np.vstack(y_list)

        base_model = LinearRegression()
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("multioutput", MultiOutputRegressor(base_model)),
            ]
        )
        self.model.fit(X_final, y_final)

    def predict(self, horizon: int) -> pd.Series:
        history_slice = self.history.iloc[-(max(self.lags) + 1) :].copy()
        features_df = generate_lagged_calendar_features(history_slice, self.lags)

        if features_df.empty:
            raise ValueError("Not enough data to compute lag-based features.")

        last_features = features_df.drop(columns=["load"]).tail(1)
        pred = self.model.predict(last_features)[0]
        return pd.Series(pred)
