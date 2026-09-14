"""
GeoMind AI - Test Suite: FastAPI Prediction Endpoint
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Integration tests for all API routes using FastAPI TestClient.
Covers: health check, model info, single prediction, batch prediction,
input validation errors, and unsupported model types.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ──────────────────────────────────────────────────────────────────────────────
# Client Fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create test client — loads models once per test module."""
    from api.main import app
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────────────
# Shared Test Data
# ──────────────────────────────────────────────────────────────────────────────

VALID_OBSERVATION = {
    "date_time": "2024-08-15T08:00:00",
    "temp": 287.5,
    "rain_1h": 0.0,
    "snow_1h": 0.0,
    "clouds_all": 40.0,
    "weather_main": "Clear",
    "holiday": "None",
    "traffic_volume": 4200.0
}

VALID_SINGLE_REQUEST = {
    "observation": VALID_OBSERVATION,
    "model_type": "xgboost"
}

VALID_BATCH_OBSERVATIONS = [
    {
        "date_time": f"2024-08-15T{hour:02d}:00:00",
        "temp": 287.5,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 40.0,
        "weather_main": "Clear",
        "holiday": "None",
        "traffic_volume": 3000.0 + hour * 100
    }
    for hour in range(15)  # 15 sequential hourly observations
]


# ──────────────────────────────────────────────────────────────────────────────
# Root & Health
# ──────────────────────────────────────────────────────────────────────────────

class TestRootAndHealth:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_project_name(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "GeoMind" in data["project"]

    def test_root_lists_endpoints(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "endpoints" in data
        assert "single_predict" in data["endpoints"]
        assert "batch_predict" in data["endpoints"]

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_models_loaded_field(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "models_loaded" in data

    def test_health_status_is_string(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["status"], str)
        assert data["status"] in ("healthy", "degraded")


# ──────────────────────────────────────────────────────────────────────────────
# Model Info
# ──────────────────────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_models_endpoint_returns_200(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_models_response_has_available_models(self, client):
        data = client.get("/models").json()
        assert "available_models" in data
        assert "xgboost" in data["available_models"]

    def test_models_response_has_metrics(self, client):
        data = client.get("/models").json()
        assert "best_model_test_mae" in data
        assert "best_model_test_r2" in data
        assert data["best_model_test_r2"] > 0.9


# ──────────────────────────────────────────────────────────────────────────────
# Single Prediction
# ──────────────────────────────────────────────────────────────────────────────

class TestSinglePrediction:
    def test_xgboost_prediction_returns_200(self, client):
        resp = client.post("/predict", json=VALID_SINGLE_REQUEST)
        assert resp.status_code == 200

    def test_prediction_response_structure(self, client):
        resp = client.post("/predict", json=VALID_SINGLE_REQUEST)
        data = resp.json()
        assert "prediction" in data
        assert "predicted_traffic_volume" in data["prediction"]
        assert "inference_time_ms" in data
        assert "status" in data

    def test_predicted_volume_is_positive(self, client):
        resp = client.post("/predict", json=VALID_SINGLE_REQUEST)
        vol = resp.json()["prediction"]["predicted_traffic_volume"]
        assert vol > 0.0

    def test_predicted_volume_is_realistic(self, client):
        """Traffic volume on an interstate should be in [0, 10000] veh/hr."""
        resp = client.post("/predict", json=VALID_SINGLE_REQUEST)
        vol = resp.json()["prediction"]["predicted_traffic_volume"]
        assert 0 <= vol <= 10000

    def test_inference_time_is_fast(self, client):
        """XGBoost inference must complete in < 500ms."""
        resp = client.post("/predict", json=VALID_SINGLE_REQUEST)
        ms = resp.json()["inference_time_ms"]
        assert ms < 500.0

    def test_model_field_reflected_in_response(self, client):
        req = {**VALID_SINGLE_REQUEST, "model_type": "xgboost"}
        resp = client.post("/predict", json=req)
        assert resp.json()["prediction"]["model_used"] == "xgboost"

    def test_random_forest_prediction_returns_200(self, client):
        req = {**VALID_SINGLE_REQUEST, "model_type": "random_forest"}
        resp = client.post("/predict", json=req)
        # May be 200 (model loaded) or 503 (model artifact missing in test env)
        assert resp.status_code in (200, 503)

    def test_invalid_model_type_returns_error(self, client):
        req = {**VALID_SINGLE_REQUEST, "model_type": "neural_ode"}
        resp = client.post("/predict", json=req)
        assert resp.status_code in (400, 422)

    def test_invalid_temp_returns_422(self, client):
        """Temperature out of range [200, 330] K → validation error."""
        obs = {**VALID_OBSERVATION, "temp": 999.0}  # Invalid
        req = {"observation": obs, "model_type": "xgboost"}
        resp = client.post("/predict", json=req)
        assert resp.status_code == 422

    def test_negative_traffic_volume_returns_422(self, client):
        obs = {**VALID_OBSERVATION, "traffic_volume": -100.0}
        req = {"observation": obs, "model_type": "xgboost"}
        resp = client.post("/predict", json=req)
        assert resp.status_code == 422

    def test_invalid_datetime_format_returns_422(self, client):
        obs = {**VALID_OBSERVATION, "date_time": "15-Aug-2024 08:00"}
        req = {"observation": obs, "model_type": "xgboost"}
        resp = client.post("/predict", json=req)
        assert resp.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        obs = {k: v for k, v in VALID_OBSERVATION.items() if k != "traffic_volume"}
        req = {"observation": obs, "model_type": "xgboost"}
        resp = client.post("/predict", json=req)
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Batch Prediction
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchPrediction:
    def test_batch_xgboost_returns_200(self, client):
        req = {"observations": VALID_BATCH_OBSERVATIONS, "model_type": "xgboost"}
        resp = client.post("/predict/batch", json=req)
        assert resp.status_code == 200

    def test_batch_response_count_matches_input(self, client):
        req = {"observations": VALID_BATCH_OBSERVATIONS, "model_type": "xgboost"}
        resp = client.post("/predict/batch", json=req)
        data = resp.json()
        assert data["total_predictions"] == len(VALID_BATCH_OBSERVATIONS)
        assert len(data["predictions"]) == len(VALID_BATCH_OBSERVATIONS)

    def test_batch_all_volumes_positive(self, client):
        req = {"observations": VALID_BATCH_OBSERVATIONS, "model_type": "xgboost"}
        resp = client.post("/predict/batch", json=req)
        for pred in resp.json()["predictions"]:
            assert pred["predicted_traffic_volume"] > 0

    def test_single_observation_batch(self, client):
        req = {"observations": [VALID_OBSERVATION], "model_type": "xgboost"}
        resp = client.post("/predict/batch", json=req)
        assert resp.status_code == 200
        assert resp.json()["total_predictions"] == 1

    def test_empty_observations_returns_422(self, client):
        req = {"observations": [], "model_type": "xgboost"}
        resp = client.post("/predict/batch", json=req)
        assert resp.status_code == 422

    def test_batch_lstm_returns_200(self, client):
        req = {"observations": VALID_BATCH_OBSERVATIONS, "model_type": "lstm"}
        resp = client.post("/predict/batch", json=req)
        # Allow 200 or 503 (model may not be loaded in test env)
        assert resp.status_code in (200, 503)


# ──────────────────────────────────────────────────────────────────────────────
# Documentation Routes
# ──────────────────────────────────────────────────────────────────────────────

class TestDocumentationRoutes:
    def test_swagger_ui_returns_200(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_returns_200(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_has_predict_route(self, client):
        data = client.get("/openapi.json").json()
        paths = data.get("paths", {})
        assert "/predict" in paths
        assert "/predict/batch" in paths
