import numpy as np
import pandas as pd
import os


class ISODataLoader:
    """
    A class to load and preprocess multiple files from a specified directory.
    Supports .csv and .xlsx files.
    """

    def __init__(self, directory):
        self.directory = directory

    def load_all(self) -> pd.DataFrame:
        all_files = [
            f
            for f in os.listdir(self.directory)
            if f.endswith(".csv") or f.endswith(".xlsx")
        ]
        data_frames = []
        for file in all_files:
            path = os.path.join(self.directory, file)
            if file.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            data_frames.append(df)
        return pd.concat(data_frames, ignore_index=True)

    def preprocess(
        self, df: pd.DataFrame, datetime_col="Date", hour_col="HR", load_col="CAISO"
    ) -> pd.DataFrame:
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df["timestamp"] = df[datetime_col] + pd.to_timedelta(df[hour_col] - 1, unit="H")

        return df
