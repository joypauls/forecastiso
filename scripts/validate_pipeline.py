# import os

from forecastiso.data_loader import ISODataLoader
from forecastiso.features import (
    FeatureManager,
    LagFeatureGenerator,
    RollingFeatureGenerator,
    CalendarFeatureGenerator,
    InteractionFeatureGenerator,
)

CAISO_DIR = "./data/caiso_hourly/"

if __name__ == "__main__":
    # data loading and preprocessing
    loader = ISODataLoader(CAISO_DIR)
    df = loader.load_and_preprocess()

    # feature generation
    fm = FeatureManager()
    fm.add_generator(LagFeatureGenerator(lags=[24, 48, 168]))
    fm.add_generator(RollingFeatureGenerator(windows=[24, 168, 720]))
    fm.add_generator(CalendarFeatureGenerator())
    fm.add_generator(InteractionFeatureGenerator())
    features_df = fm.generate_features(df)

    print(features_df.columns)
    # print(features_df.head())
    print(features_df.shape)

    # # save preprocessed data
    # output_file = os.path.join(CAISO_DIR, "preprocessed_hourly_load.pkl")
    # df.to_pickle(output_file)
