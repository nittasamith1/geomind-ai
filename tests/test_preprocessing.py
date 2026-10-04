import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.feature_engineering import engineer_features
from src.data_preprocessing import fit_preprocessor, transform, get_feature_columns


def make_hourly_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "date_time":      dates,
        "traffic_volume": rng.integers(500, 6000, size=n).astype(float),
        "temp":           rng.uniform(250, 310, size=n),
        "rain_1h":        rng.uniform(0, 5, size=n),
        "snow_1h":        0.0,
        "clouds_all":     rng.uniform(0, 100, size=n),
        "weather_main":   rng.choice(["Clear", "Clouds", "Rain"], size=n),
        "holiday":        "None",
    })


@pytest.fixture
def train_feat():
    return engineer_features(make_hourly_df(n=100, seed=1), is_training=True)


@pytest.fixture
def val_feat():
    return engineer_features(make_hourly_df(n=60, seed=2), is_training=True)


class TestPreprocessor:
    def test_fit_returns_pipeline_and_names(self, train_feat):
        pipeline, names = fit_preprocessor(train_feat, save=False)
        assert pipeline is not None
        assert len(names) > 0

    def test_transform_shape_matches(self, train_feat):
        pipeline, names = fit_preprocessor(train_feat, save=False)
        X, y = transform(pipeline, train_feat)
        assert X.shape[0] == len(train_feat)
        assert X.shape[1] == len(names)

    def test_transform_target_not_none(self, train_feat):
        pipeline, names = fit_preprocessor(train_feat, save=False)
        _, y = transform(pipeline, train_feat)
        assert y is not None
        assert len(y) == len(train_feat)

    def test_val_uses_train_feature_count(self, train_feat, val_feat):
        pipeline, names = fit_preprocessor(train_feat, save=False)
        X_tr, _ = transform(pipeline, train_feat)
        X_va, _ = transform(pipeline, val_feat)
        assert X_tr.shape[1] == X_va.shape[1]

    def test_no_nan_in_output(self, train_feat):
        pipeline, names = fit_preprocessor(train_feat, save=False)
        X, y = transform(pipeline, train_feat)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_traffic_volume_not_in_features(self, train_feat):
        pipeline, names = fit_preprocessor(train_feat, save=False)
        assert "traffic_volume" not in names
        assert "traffic_volume_target" not in names
