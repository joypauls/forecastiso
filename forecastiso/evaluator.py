import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def mean_error(y_true, y_pred):
    """Mean Error (bias)"""
    return np.mean(y_pred - y_true)


class Evaluator:
    """
    Evaluator for electricity load forecasting models using time series cross-validation.

    Supports periodic model retraining and customizable metrics.
    """

    def __init__(
        self,
        retrain_frequency: int = 7,
        train_days: int = 730,
        horizon: int = 24,
        metrics: Optional[Dict[str, Callable]] = None,
        skip_dst: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            retrain_frequency: Retrain model every N days
            train_days: Number of days to use for training
            horizon: Prediction horizon in hours
            metrics: Dict of metric_name -> metric_function. If None, uses default metrics
            skip_dst: Whether to skip daylight saving time transition days
            verbose: Whether to print progress messages
        """
        self.retrain_frequency = retrain_frequency
        self.train_days = train_days
        self.horizon = horizon
        self.skip_dst = skip_dst
        self.verbose = verbose

        # Default metrics
        default_metrics = {
            "mape": mean_absolute_percentage_error,
            "mae": mean_absolute_error,
            "rmse": root_mean_squared_error,
            "me": mean_error,
        }

        self.metrics = metrics if metrics is not None else default_metrics

        # Results storage
        self.reset_results()

    def reset_results(self):
        """Reset stored results"""
        self.predictions = []
        self.true_values = []
        self.prediction_dates = []
        self.metric_values = {name: [] for name in self.metrics.keys()}
        self.model = None

    def _should_retrain(self, day_index: int) -> bool:
        """Determine if model should be retrained on this day"""
        return day_index % self.retrain_frequency == 0

    def _is_dst_day(
        self, date: pd.Timestamp, day_before_df: pd.DataFrame, predict_df: pd.DataFrame
    ) -> bool:
        """Check if this is a DST transition day to potentially skip"""
        if not self.skip_dst:
            return False

        # Check for DST months and irregular hour counts
        is_dst_month = date.month in [3, 11]
        irregular_hours = day_before_df.shape[0] != 24 or len(predict_df) != 24

        return is_dst_month and irregular_hours

    def _validate_features(self, latest_row: pd.Series, predict_date: pd.Timestamp):
        """Validate feature consistency"""
        if "target_dow" in latest_row:
            if latest_row["target_dow"] != predict_date.dayofweek:
                if self.verbose:
                    print(
                        f"Warning: target_dow {latest_row['target_dow']} does not match predict dow {predict_date.dayofweek}"
                    )

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all metrics for a prediction"""
        results = {}
        for name, metric_func in self.metrics.items():
            try:
                results[name] = metric_func(y_true, y_pred)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not calculate {name}: {e}")
                results[name] = np.nan
        return results

    def evaluate(
        self,
        model_class: Any,
        model_kwargs: Dict[str, Any],
        features_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        first_test_date: str,
        last_test_date: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a model using time series cross-validation.

        Args:
            model_class: Model class to instantiate (e.g., XGBForecaster)
            model_kwargs: Kwargs to pass to model constructor
            features_df: DataFrame with features and target, indexed by datetime
            target_col: Name of target column
            feature_cols: List of feature column names
            first_test_date: Start date for testing period
            last_test_date: End date for testing period

        Returns:
            Dict containing predictions, true values, dates, and metric summaries
        """
        self.reset_results()

        first_date = pd.to_datetime(first_test_date)
        last_date = pd.to_datetime(last_test_date)
        test_days = (last_date - first_date).days

        for i in range(test_days):
            cur_predict_date = first_date + pd.Timedelta(days=i)
            cur_day_before_date = first_date + pd.Timedelta(days=i - 1)

            # Retrain model if needed
            if self._should_retrain(i):
                if self.verbose:
                    print(f"Training model for day {i} at {cur_predict_date}")

                train_start_date = cur_day_before_date - pd.Timedelta(
                    days=self.train_days
                )
                train_end_date = (
                    cur_day_before_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
                )
                train_df = features_df[train_start_date:train_end_date]

                self.model = model_class(
                    target_col=target_col, feature_cols=feature_cols, **model_kwargs
                )
                self.model.fit(train_df)

            # Get prediction data
            predict_start = cur_predict_date
            predict_end = (
                cur_predict_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
            )
            predict_df = features_df[predict_start:predict_end]

            # Get day-before data for features
            day_before_start = cur_day_before_date
            day_before_end = (
                cur_day_before_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
            )
            day_before_df = features_df[day_before_start:day_before_end]

            # Skip DST days if configured
            if self._is_dst_day(cur_predict_date, day_before_df, predict_df):
                if self.verbose:
                    print(
                        f"Warning: Skipping prediction for {cur_predict_date} due to unsupported DST."
                    )
                continue

            # Get true values and latest features
            y_true = predict_df[target_col].reset_index(drop=True)
            latest_row = day_before_df.iloc[-1]

            # Validate features
            self._validate_features(latest_row, cur_predict_date)

            # Make prediction
            y_pred = self.model.predict(horizon=self.horizon, input_features=latest_row)

            # Store results
            self.predictions.append(y_pred)
            self.true_values.append(y_true)
            self.prediction_dates.append(cur_predict_date)

            # Calculate and store metrics
            day_metrics = self._calculate_metrics(y_true, y_pred)
            for name, value in day_metrics.items():
                self.metric_values[name].append(value)

        if self.verbose:
            print(f"Evaluation complete. Processed {len(self.predictions)} days.")

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get evaluation results"""
        results = {
            "predictions": self.predictions,
            "true_values": self.true_values,
            "prediction_dates": self.prediction_dates,
            "daily_metrics": self.metric_values.copy(),
            "summary_metrics": {},
        }

        # Calculate summary statistics for each metric
        for name, values in self.metric_values.items():
            if values:  # Only if we have values
                clean_values = [v for v in values if not np.isnan(v)]
                if clean_values:
                    results["summary_metrics"][name] = {
                        "mean": np.mean(clean_values),
                        "std": np.std(clean_values),
                        "median": np.median(clean_values),
                        "min": np.min(clean_values),
                        "max": np.max(clean_values),
                    }

        return results

    def print_summary(self):
        """Print summary of evaluation results"""
        results = self.get_results()
        print(f"\nEvaluation Summary ({len(self.predictions)} days):")
        print("-" * 50)

        for metric_name, stats in results["summary_metrics"].items():
            print(f"{metric_name.upper()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print()
