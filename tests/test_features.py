"""
GeoMind AI - Test Suite: Feature Engineering
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Tests temporal correctness, lag validity, rolling statistics,
target construction, and pipeline integration.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.feature_engineering import (
    create_target_variable,
    add_temporal_features,
    add_autoregressive_lags,
    add_rolling_features,
    engineer_features,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def continuous_hourly_df():
    """24 continuous hourly observations — no gaps."""
    dates = pd.date_range("2024-01-01 00:00", periods=48, freq="h")
    np.random.seed(42)
    traffic = np.random.randint(1000, 5000, size=48).astype(float)
    return pd.DataFrame({
        "date_time": dates,
        "traffic_volume": traffic,
        "temp": 280.0,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 20.0,
        "weather_main": "Clear",
        "holiday": "None",
    })


@pytest.fixture
def gapped_df():
    """DataFrame with a 2-hour gap in the middle — tests temporal discontinuity handling."""
    dates1 = pd.date_range("2024-01-01 00:00", periods=12, freq="h")
    dates2 = pd.date_range("2024-01-01 14:00", periods=12, freq="h")  # 2-hour gap at 12:00
    dates = dates1.append(dates2)
    traffic = np.ones(24, dtype=float) * 3000.0
    return pd.DataFrame({
        "date_time": dates,
        "traffic_volume": traffic,
        "temp": 275.0,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 50.0,
        "weather_main": "Clouds",
        "holiday": "None",
    })


# ──────────────────────────────────────────────────────────────────────────────
# Target Variable Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTargetVariable:
    def test_target_constructed(self, continuous_hourly_df):
        df = create_target_variable(continuous_hourly_df)
        assert "future_traffic_volume" in df.columns

    def test_last_row_target_is_nan(self, continuous_hourly_df):
        """Last observation has no t+1 successor → target must be NaN."""
        df = create_target_variable(continuous_hourly_df)
        assert pd.isna(df["future_traffic_volume"].iloc[-1])

    def test_continuous_rows_have_valid_target(self, continuous_hourly_df):
        df = create_target_variable(continuous_hourly_df)
        # All but last row should have valid targets
        assert df["future_traffic_volume"].iloc[:-1].notna().all()

    def test_target_equals_next_row_traffic(self, continuous_hourly_df):
        """Verify future_traffic_volume[i] == traffic_volume[i+1]."""
        df = create_target_variable(continuous_hourly_df)
        for i in range(len(df) - 1):
            assert df["future_traffic_volume"].iloc[i] == df["traffic_volume"].iloc[i + 1]

    def test_gap_rows_have_nan_target(self, gapped_df):
        """Row before temporal gap must have NaN target (no valid t+1)."""
        df = create_target_variable(gapped_df)
        # Row at index 11 (last before gap) must have NaN target
        assert pd.isna(df["future_traffic_volume"].iloc[11])


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Feature Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_all_temporal_columns_created(self, continuous_hourly_df):
        df = add_temporal_features(continuous_hourly_df)
        expected = ["hour", "day_of_week", "month", "year", "is_weekend",
                    "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week",
                    "is_rush_hour", "is_holiday"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_cyclical_hour_range(self, continuous_hourly_df):
        df = add_temporal_features(continuous_hourly_df)
        assert df["sin_hour"].between(-1.0, 1.0).all()
        assert df["cos_hour"].between(-1.0, 1.0).all()

    def test_is_weekend_binary(self, continuous_hourly_df):
        df = add_temporal_features(continuous_hourly_df)
        assert df["is_weekend"].isin([0, 1]).all()

    def test_is_rush_hour_binary(self, continuous_hourly_df):
        df = add_temporal_features(continuous_hourly_df)
        assert df["is_rush_hour"].isin([0, 1]).all()

    def test_holiday_flag_from_none_string(self, continuous_hourly_df):
        df = add_temporal_features(continuous_hourly_df)
        # All holidays are "None" → is_holiday should be 0
        assert (df["is_holiday"] == 0).all()

    def test_holiday_flag_from_real_holiday(self, continuous_hourly_df):
        df = continuous_hourly_df.copy()
        df.loc[0, "holiday"] = "Christmas Day"
        df = add_temporal_features(df)
        assert df.loc[0, "is_holiday"] == 1

    def test_cyclical_identity(self, continuous_hourly_df):
        """sin²(x) + cos²(x) == 1 for all hours."""
        df = add_temporal_features(continuous_hourly_df)
        identity = df["sin_hour"] ** 2 + df["cos_hour"] ** 2
        np.testing.assert_allclose(identity.values, 1.0, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Lag Feature Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag_columns_created(self, continuous_hourly_df):
        df = add_autoregressive_lags(continuous_hourly_df)
        for k in [1, 2, 3, 6, 12, 24]:
            assert f"traffic_lag_{k}" in df.columns

    def test_lag1_equals_previous_row(self, continuous_hourly_df):
        """traffic_lag_1[i] == traffic_volume[i-1] for continuous data."""
        df = add_autoregressive_lags(continuous_hourly_df)
        for i in range(1, len(df)):
            assert df["traffic_lag_1"].iloc[i] == df["traffic_volume"].iloc[i - 1]

    def test_lag_boundary_is_nan(self, continuous_hourly_df):
        """First k rows cannot have lag_k → must be NaN."""
        df = add_autoregressive_lags(continuous_hourly_df)
        assert pd.isna(df["traffic_lag_1"].iloc[0])
        assert pd.isna(df["traffic_lag_6"].iloc[5])

    def test_gap_produces_nan_lag(self, gapped_df):
        """Lag across temporal gap must yield NaN."""
        df = add_autoregressive_lags(gapped_df)
        # First row after the gap (index 12) cannot have lag_1 (gap exists)
        assert pd.isna(df["traffic_lag_1"].iloc[12])


# ──────────────────────────────────────────────────────────────────────────────
# Rolling Feature Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRollingFeatures:
    def test_rolling_columns_created(self, continuous_hourly_df):
        df = add_rolling_features(continuous_hourly_df)
        for w in [3, 6, 24]:
            assert f"rolling_mean_{w}h" in df.columns
            assert f"rolling_std_{w}h" in df.columns

    def test_rolling_mean_non_negative(self, continuous_hourly_df):
        df = add_rolling_features(continuous_hourly_df)
        assert (df["rolling_mean_3h"].dropna() >= 0).all()

    def test_rolling_std_non_negative(self, continuous_hourly_df):
        df = add_rolling_features(continuous_hourly_df)
        assert (df["rolling_std_3h"].dropna() >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Full Pipeline Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEngineerFeaturesPipeline:
    def test_pipeline_runs_without_error(self, continuous_hourly_df):
        df = engineer_features(continuous_hourly_df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_no_nan_in_target_after_pipeline(self, continuous_hourly_df):
        df = engineer_features(continuous_hourly_df)
        assert df["future_traffic_volume"].notna().all()

    def test_no_nan_in_lag1_after_pipeline(self, continuous_hourly_df):
        df = engineer_features(continuous_hourly_df)
        assert df["traffic_lag_1"].notna().all()

    def test_expected_feature_columns_present(self, continuous_hourly_df):
        df = engineer_features(continuous_hourly_df)
        required_cols = [
            "future_traffic_volume", "sin_hour", "cos_hour",
            "traffic_lag_1", "rolling_mean_3h", "is_rush_hour"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing: {col}"

    def test_row_count_shrinks_after_dropna(self, continuous_hourly_df):
        """Pipeline drops boundary rows → output must be smaller than input."""
        df = engineer_features(continuous_hourly_df)
        assert len(df) < len(continuous_hourly_df)

    def test_output_dtypes_are_numeric(self, continuous_hourly_df):
        """All engineered numerical features must be float/int (no object columns except datetime/categorical)."""
        df = engineer_features(continuous_hourly_df)
        non_numeric = [
            col for col in df.columns
            if col not in ["date_time", "weather_main", "holiday", "weather_description"]
            and df[col].dtype == object
        ]
        assert len(non_numeric) == 0, f"Non-numeric columns found: {non_numeric}"
