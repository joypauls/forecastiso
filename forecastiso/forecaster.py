import pandas as pd
from abc import ABC, abstractmethod


class Forecaster(ABC):
    def __init__(self, name="UnnamedForecaster"):
        self.name = name

    @abstractmethod
    def fit(self, history: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        pass
