import pandas as pd

from forecastiso.evaluator import Evaluator


class ForecastManager:
    def __init__(self, forecasters):
        self.forecasters = forecasters
        self.evaluator = Evaluator()
        self.predictions = {}

    def fit_all(self, train_df):
        for forecaster in self.forecasters:
            forecaster.fit(train_df)

    def predict_all(self, horizon=24):
        for forecaster in self.forecasters:
            self.predictions[forecaster.name] = forecaster.predict(horizon)
        return self.predictions

    def evaluate_all(self, test_df):
        for name, preds in self.predictions.items():
            self.evaluator.evaluate(test_df["load"].reset_index(drop=True), preds, name)
        return self.evaluator.summary()

    def walk_forward_all(self, df: pd.DataFrame, start_offset: int = 7, steps: int = 7):
        for forecaster in self.forecasters:
            self.evaluator.walk_forward(df, forecaster, start_offset, steps)

    def summary(self):
        return self.evaluator.summary()
