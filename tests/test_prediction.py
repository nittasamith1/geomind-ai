import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


VALID_OBS = {
    "date_time":      "2024-08-15T08:00:00",
    "temp":           287.5,
    "rain_1h":        0.0,
    "snow_1h":        0.0,
    "clouds_all":     40.0,
    "weather_main":   "Clear",
    "holiday":        "None",
}

VALID_REQUEST = {"observation": VALID_OBS, "model_type": "xgboost"}


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_has_models_loaded(self, client):
        data = client.get("/health").json()
        assert "models_loaded" in data


class TestModels:
    def test_models_returns_200(self, client):
        assert client.get("/models").status_code == 200

    def test_models_has_xgboost(self, client):
        data = client.get("/models").json()
        assert "xgboost" in data["available_models"]


class TestPredict:
    def test_xgboost_returns_200(self, client):
        resp = client.post("/predict", json=VALID_REQUEST)
        assert resp.status_code == 200

    def test_response_has_prediction(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert "predicted_traffic_volume" in data
        assert "traffic_level" in data

    def test_prediction_is_positive(self, client):
        vol = client.post("/predict", json=VALID_REQUEST).json()["predicted_traffic_volume"]
        assert vol >= 0.0

    def test_prediction_is_realistic(self, client):
        vol = client.post("/predict", json=VALID_REQUEST).json()["predicted_traffic_volume"]
        assert 0 <= vol <= 10000

    def test_random_forest_works(self, client):
        req = {**VALID_REQUEST, "model_type": "random_forest"}
        assert client.post("/predict", json=req).status_code in (200, 503)

    def test_ridge_regression_works(self, client):
        req = {**VALID_REQUEST, "model_type": "ridge_regression"}
        assert client.post("/predict", json=req).status_code in (200, 503)

    def test_invalid_model_returns_error(self, client):
        req = {**VALID_REQUEST, "model_type": "lstm"}
        assert client.post("/predict", json=req).status_code == 422

    def test_invalid_temp_returns_422(self, client):
        obs = {**VALID_OBS, "temp": 999.0}
        assert client.post("/predict", json={"observation": obs, "model_type": "xgboost"}).status_code == 422

    def test_bad_datetime_returns_422(self, client):
        obs = {**VALID_OBS, "date_time": "not-a-date"}
        assert client.post("/predict", json={"observation": obs, "model_type": "xgboost"}).status_code == 422

    def test_no_traffic_volume_in_observation_succeeds(self, client):
        # Explicit test proving observation does NOT need traffic_volume
        assert "traffic_volume" not in VALID_OBS
        resp = client.post("/predict", json={"observation": VALID_OBS, "model_type": "xgboost"})
        assert resp.status_code == 200

    def test_inference_time_present(self, client):
        data = client.post("/predict", json=VALID_REQUEST).json()
        assert "inference_ms" in data
        assert data["inference_ms"] < 2000


class TestDocs:
    def test_swagger_returns_200(self, client):
        assert client.get("/docs").status_code == 200

    def test_openapi_has_predict_route(self, client):
        paths = client.get("/openapi.json").json().get("paths", {})
        assert "/predict" in paths
