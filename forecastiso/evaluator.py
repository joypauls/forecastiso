import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from forecastiso.forecaster import Forecaster


class Evaluator:
    def __init__(self):
        self.results = []

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, model_name: str):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        self.results.append({"model": model_name, "MAE": mae, "RMSE": rmse})
        return {"MAE": mae, "RMSE": rmse}

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def walk_forward(
        self,
        df: pd.DataFrame,
        forecaster: Forecaster,
        start_offset: int = 7,
        steps: int = 7,
    ):
        horizon = 24
        for i in range(steps):
            split = (start_offset + i) * 24
            train_df = df.iloc[:split]
            test_df = df.iloc[split : split + horizon]
            forecaster.fit(train_df)
            y_pred = forecaster.predict(horizon=horizon)
            self.evaluate(
                test_df["load"].reset_index(drop=True),
                y_pred,
                f"{forecaster.name}_t{i}",
            )
