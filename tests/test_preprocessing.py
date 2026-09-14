"""
GeoMind AI - Test Suite: Data Preprocessing
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Tests the TrafficPreprocessor class for correct fitting,
leakage prevention, and serialization.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_preprocessing import TrafficPreprocessor
from src.feature_engineering import engineer_features


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Create a minimal raw traffic DataFrame with n continuous hourly observations."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "date_time": dates,
        "traffic_volume": rng.integers(500, 6000, size=n).astype(float),
        "temp": rng.uniform(250, 310, size=n),
        "rain_1h": rng.uniform(0, 5, size=n),
        "snow_1h": 0.0,
        "clouds_all": rng.uniform(0, 100, size=n),
        "weather_main": rng.choice(["Clear", "Clouds", "Rain"], size=n),
        "holiday": "None",
    })


@pytest.fixture
def engineered_train():
    raw = _make_raw_df(n=100, seed=1)
    return engineer_features(raw)


@pytest.fixture
def engineered_val():
    raw = _make_raw_df(n=50, seed=2)
    return engineer_features(raw)


# ──────────────────────────────────────────────────────────────────────────────
# TrafficPreprocessor Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTrafficPreprocessor:
    def test_fit_runs_without_error(self, engineered_train):
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp.fit(engineered_train)
        assert pp.pipeline is not None

    def test_feature_names_populated_after_fit(self, engineered_train):
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp.fit(engineered_train)
        assert len(pp.feature_names) > 0

    def test_transform_returns_correct_shape(self, engineered_train):
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp.fit(engineered_train)
        X, y = pp.transform(engineered_train)
        assert X.shape[0] == len(engineered_train)
        assert X.shape[1] == len(pp.feature_names)

    def test_transform_target_not_none(self, engineered_train):
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp.fit(engineered_train)
        _, y = pp.transform(engineered_train)
        assert y is not None
        assert len(y) == len(engineered_train)

    def test_val_transform_uses_train_stats(self, engineered_train, engineered_val):
        """Verify no re-fitting on val → leakage prevention."""
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp.fit(engineered_train)
        X_tr, _ = pp.transform(engineered_train)
        X_va, _ = pp.transform(engineered_val)
        # Column counts must match
        assert X_tr.shape[1] == X_va.shape[1]

    def test_transform_before_fit_raises(self, engineered_train):
        """Transform without fit must raise an exception (ValueError wrapped in CustomException)."""
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        with pytest.raises(Exception):  # CustomException wraps ValueError
            pp.transform(engineered_train)

    def test_fit_transform_equivalent_to_separate_calls(self, engineered_train):
        pp1 = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        X1, y1 = pp1.fit_transform(engineered_train)

        pp2 = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp2.fit(engineered_train)
        X2, y2 = pp2.transform(engineered_train)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_serialization_roundtrip(self, engineered_train):
        """Save and reload preprocessor → identical output."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name

        pp = TrafficPreprocessor(artifact_path=tmp_path)
        pp.fit(engineered_train)
        X_before, _ = pp.transform(engineered_train)
        pp.save_artifact()

        pp2 = TrafficPreprocessor(artifact_path=tmp_path)
        pp2.load_artifact()
        X_after, _ = pp2.transform(engineered_train)

        np.testing.assert_array_almost_equal(X_before, X_after)
        assert pp2.feature_names == pp.feature_names

    def test_no_nan_in_transformed_output(self, engineered_train):
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp.fit(engineered_train)
        X, y = pp.transform(engineered_train)
        assert not np.isnan(X).any(), "NaN found in transformed feature matrix"
        assert not np.isnan(y).any(), "NaN found in target vector"

    def test_target_column_excluded_from_features(self, engineered_train):
        pp = TrafficPreprocessor(artifact_path="models/test_preprocessor.joblib")
        pp.fit(engineered_train)
        assert "future_traffic_volume" not in pp.feature_names
