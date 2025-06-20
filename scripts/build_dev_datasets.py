import numpy as np
import pandas as pd
import os

from forecastiso.data_loader import ISODataLoader

CAISO_DIR = "./data/caiso_hourly/"

if __name__ == "__main__":
    loader = ISODataLoader(CAISO_DIR)
    df = loader.load_and_preprocess()

    # save preprocessed data
    output_file = os.path.join(CAISO_DIR, "preprocessed_hourly_load.pkl")
    df.to_pickle(output_file)

    # print(df.head())
    # print(f"Data shape: {df.shape}")
