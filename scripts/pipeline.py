"""
Minimal script used to validate full forecasting pipeline and for one-off runs.
"""

import logging
import pandas as pd
import os
import pickle

from forecastiso.data_loader import ISODataLoader
from forecastiso.features import (
    FeatureManager,
    LagFeatureGenerator,
    RollingFeatureGenerator,
    CalendarFeatureGenerator,
    WindowFeatureGenerator,
)
from forecastiso.forecasters import (
    XGBForecaster,
    ARIMAForecaster,
    YesterdayForecaster,
    LastWeekForecaster,
    RollingMeanForecaster,
)
from forecastiso.evaluator import Evaluator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("| %(asctime)s | %(name)s | %(levelname)s | %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# RUN_NAME = "pipeline_validation"
RUN_NAME = "caiso_2024"
DATA_DIR = "./data/caiso_hourly/"
OUTPUT_DIR = "./output/"
TARGET_COL = "load"
FIRST_TEST_DATE = "2024-01-01"
LAST_TEST_DATE = "2024-12-31"
TEST_DAYS = (pd.to_datetime(LAST_TEST_DATE) - pd.to_datetime(FIRST_TEST_DATE)).days
TRAIN_DAYS = 730
HORIZON = 24

if not os.path.exists(f"{OUTPUT_DIR}/{RUN_NAME}"):
    os.makedirs(f"{OUTPUT_DIR}/{RUN_NAME}")


def _get_feature_columns() -> list[str]:
    windowed_feature_cols = [f"load_window_{24}_{i}" for i in range(0, 24)]
    return [
        "load_lag_24",
        "load_lag_48",
        "load_lag_168",
        "load_rolling_mean_24",
        "load_rolling_min_24",
        "load_rolling_max_24",
        "load_rolling_std_24",
        "load_rolling_mean_168",
        "load_rolling_min_168",
        "load_rolling_max_168",
        "load_rolling_std_168",
        "load_rolling_mean_720",
        "load_rolling_min_720",
        "load_rolling_max_720",
        "load_rolling_std_720",
        "hour",
        "target_month",
        "target_dow",
        "target_doy",
        "target_is_holiday",
        "target_day_before_holiday",
        "target_is_weekend",
        "target_day",
        "target_quarter",
        "target_year",
    ] + windowed_feature_cols


def _dict_to_pkl(d: dict, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(d, f)


def _evaluate_and_save_model(
    evaluator: Evaluator,
    model_class,
    model_kwargs: dict,
    features_df: pd.DataFrame,
    model_name: str,
):
    evaluator.reset_results()

    print("---" * 20)
    logger.info(f"Evaluating {model_class.__name__}")
    results = evaluator.evaluate(
        model_class=model_class,
        model_kwargs=model_kwargs,
        features_df=features_df,
        target_col=TARGET_COL,
        first_test_date=FIRST_TEST_DATE,
        last_test_date=LAST_TEST_DATE,
    )
    logger.info(f"MAPE: {results['summary_metrics']['mape']['mean']}")
    _dict_to_pkl(
        results,
        f"{OUTPUT_DIR}/{RUN_NAME}/results_{model_name}.pkl",
    )
    return results


if __name__ == "__main__":
    logger.info(f"Loading datasets from {DATA_DIR}")
    loader = ISODataLoader(DATA_DIR)
    df = loader.load_and_preprocess()
    df = df[df["area"] == "caiso"]
    df.to_pickle(f"{OUTPUT_DIR}/{RUN_NAME}/raw.pkl")

    print("Loaded data time period:")
    print(f"  Min: {df['datetime'].min()}")
    print(f"  Max: {df['datetime'].max()}")

    logger.info("Generating features")
    fm = FeatureManager()
    fm.add_generator(LagFeatureGenerator(column=TARGET_COL, lags=[24, 48, 168]))
    fm.add_generator(RollingFeatureGenerator(column=TARGET_COL, windows=[24, 168, 720]))
    fm.add_generator(CalendarFeatureGenerator())
    fm.add_generator(WindowFeatureGenerator(column=TARGET_COL, window_sizes=[24]))
    features_df = fm.generate_features(df)
    features_df.to_pickle(f"{OUTPUT_DIR}/{RUN_NAME}/features.pkl")

    print("Generated features time period:")
    print(f"  Min: {features_df.index.min()}")
    print(f"  Max: {features_df.index.max()}")

    daily_evaluator = Evaluator(
        retrain_frequency=1,
        train_days=TRAIN_DAYS,
        horizon=HORIZON,
        verbose=False,
        skip_dst=True,
    )
    weekly_evaluator = Evaluator(
        retrain_frequency=7,
        train_days=TRAIN_DAYS,
        horizon=HORIZON,
        verbose=False,
        skip_dst=True,
    )
    models_config = [
        (YesterdayForecaster, {}, "yesterday", daily_evaluator),
        (LastWeekForecaster, {}, "lastweek", daily_evaluator),
        (RollingMeanForecaster, {}, "rollingmean", daily_evaluator),
        (ARIMAForecaster, {}, "arima", daily_evaluator),
        (
            XGBForecaster,
            {"feature_cols": _get_feature_columns()},
            "xgb",
            weekly_evaluator,
        ),
    ]
    # do something with results?
    results = {}
    for model_class, model_kwargs, model_name, model_evaluator in models_config:
        results[model_name] = _evaluate_and_save_model(
            evaluator=model_evaluator,
            model_class=model_class,
            model_kwargs=model_kwargs,
            features_df=features_df,
            model_name=model_name,
        )
