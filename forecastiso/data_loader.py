import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class ISODataLoader:
    """
    A class to load and preprocess multiple files from a specified directory.
    Supports .csv and .xlsx files.
    Assumes a fairly standard structure for the files, with a datetime column,
    an hour column, and one or several load (demand) columns corresponding to
    system-wide or subregion load.
    """

    def __init__(self, directory: str):
        self.directory = directory

    def load_batch(self) -> pd.DataFrame:
        """Load all .csv and .xlsx files from the specified directory into a single DataFrame."""

        # identify files in the directory
        all_files = [
            f
            for f in os.listdir(self.directory)
            if f.endswith(".csv") or f.endswith(".xlsx")
        ]

        if not all_files:
            logger.warning(f"No valid files found in directory: {self.directory}")
            return pd.DataFrame()

        # iterate through files and combine
        df_list = []
        for file in all_files:
            path = os.path.join(self.directory, file)
            if file.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    def preprocess(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        hour_col: str = "hr",
        area_col: str = "caiso",
    ) -> pd.DataFrame:
        """Some standardization steps to get rid of individual quirks of the different ISO files."""
        df = df.copy()

        # convert all columns to lower case first
        df.columns = df.columns.str.lower()

        # check if the required columns are present
        required_columns = {date_col, hour_col, area_col}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")

        # subset to only the relevant columns and filter
        df = df[list(required_columns)]
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # 1 single normalized datetime column
        df[date_col] = pd.to_datetime(df[date_col])
        df["datetime"] = df[date_col] + pd.to_timedelta(df[hour_col] - 1, unit="h")
        df.drop(columns=[date_col, hour_col], inplace=True, errors="ignore")

        # convert to a long format to eventually support multiple regions
        df_long = df.melt(id_vars=["datetime"], var_name="area", value_name="load")

        return (
            df_long[["datetime", "area", "load"]]
            .sort_values(by="datetime")
            .reset_index(drop=True)
        )

    def load_and_preprocess(
        self,
        date_col: str = "date",
        hour_col: str = "hr",
        area_col: str = "caiso",
    ) -> pd.DataFrame:
        """Load and preprocess the data from the specified directory."""
        df = self.load_batch()
        return self.preprocess(df, date_col, hour_col, area_col)
