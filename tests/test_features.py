import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.feature_engineering import engineer_features


def make_hourly_df(n=60, seed=42):
    dates = pd.date_range("2024-01-01 00:00", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date_time":      dates,
        "traffic_volume": rng.integers(500, 6000, size=n).astype(float),
        "temp":           280.0,
        "rain_1h":        0.0,
        "snow_1h":        0.0,
        "clouds_all":     20.0,
        "weather_main":   "Clear",
        "holiday":        "None",
    })


@pytest.fixture
def df():
    return make_hourly_df(n=60)


class TestEngineerFeatures:
    def test_returns_dataframe(self, df):
        result = engineer_features(df, is_training=True)
        assert isinstance(result, pd.DataFrame)

    def test_target_present_in_training(self, df):
        result = engineer_features(df, is_training=True)
        assert result["traffic_volume_target"].notna().all()

    def test_inference_mode_without_traffic_volume(self, df):
        inference_df = df.drop(columns=["traffic_volume"])
        result = engineer_features(inference_df, is_training=False)
        assert isinstance(result, pd.DataFrame)
        assert "traffic_volume_target" not in result.columns

    def test_expected_columns_present(self, df):
        result = engineer_features(df, is_training=True)
        for col in ["sin_hour", "cos_hour", "sin_dow", "cos_dow",
                    "is_rush_hour", "is_weekend", "is_holiday",
                    "temp_celsius", "weather_severity"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_cyclical_encoding_valid(self, df):
        result = engineer_features(df, is_training=True)
        identity = result["sin_hour"] ** 2 + result["cos_hour"] ** 2
        np.testing.assert_allclose(identity.values, 1.0, atol=1e-6)

    def test_is_weekend_binary(self, df):
        result = engineer_features(df, is_training=True)
        assert result["is_weekend"].isin([0, 1]).all()

    def test_is_rush_hour_binary(self, df):
        result = engineer_features(df, is_training=True)
        assert result["is_rush_hour"].isin([0, 1]).all()
