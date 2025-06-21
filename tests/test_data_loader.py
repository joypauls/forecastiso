import os
import pandas as pd

from forecastiso.data_loader import ISODataLoader

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "hourly_batch")


def test_load_batch():
    """Test that the loader can load data from the test files."""
    loader = ISODataLoader(TEST_DATA_DIR)
    df = loader.load_batch()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_preprocess():
    """Test the preprocessing of loaded data."""
    loader = ISODataLoader(TEST_DATA_DIR)
    raw_df = loader.load_batch()
    processed_df = loader.preprocess(raw_df)

    assert "datetime" in processed_df.columns
    assert "area" in processed_df.columns
    assert "load" in processed_df.columns
    assert len(processed_df.columns) == 3
    assert pd.api.types.is_datetime64_dtype(processed_df["datetime"])

    assert processed_df["datetime"].is_unique
    assert processed_df["datetime"].is_monotonic_increasing

    num_days = (
        processed_df["datetime"].max() - processed_df["datetime"].min()
    ).days + 1
    # need to account for daylight savings since these test files include a day with the change
    assert len(processed_df) == (num_days * 24) - 1


def test_load_and_preprocess():
    """Test the combined load and preprocess functionality."""
    loader = ISODataLoader(TEST_DATA_DIR)
    df = loader.load_and_preprocess()

    # basic conditions, since we already tested the core methods above
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "datetime" in df.columns
    assert "load" in df.columns
    assert "area" in df.columns
    assert len(df.columns) == 3
    assert df["datetime"].is_monotonic_increasing
