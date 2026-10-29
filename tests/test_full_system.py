"""
Comprehensive End-to-End System Test for GeoMind.

Tests 100% of the project's cloud capabilities:
1. Health & Model status
2. Single prediction inference
3. Multi-model compare (/predict/compare)
4. Telemetry logging & Analytics aggregation (/analytics)
5. Firebase Storage status (/storage/status)
6. Model synchronization to Firebase Storage (/storage/sync-models)
7. Querying stored artifacts in Firebase Storage (/storage/models)
8. CSV telemetry batch export to Firebase Storage (/storage/export-predictions)
9. Artifact downloading via Storage endpoint (/storage/download/...)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_system():
    print("=" * 70)
    print("  GeoMind Cloud & ML End-to-End Verification")
    print("=" * 70)

    # 1. Health check
    print("\n[1/8] Testing /health ...")
    r = client.get("/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health_data = r.json()
    print(f"      Status: {health_data['status']} | Version: {health_data['api_version']}")
    print(f"      Models loaded: {health_data['models_loaded']}")
    assert health_data["status"] == "healthy"

    # 2. Prediction
    print("\n[2/8] Testing /predict (Single Inference) ...")
    payload = {
        "observation": {
            "date_time": "2026-10-15T08:00:00",
            "temp": 288.15,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 20.0,
            "weather_main": "Clear",
            "road_condition": "Dry / Normal",
            "holiday": "None",
        },
        "model_type": "xgboost"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred_data = r.json()
    vol = pred_data["predicted_traffic_volume"]
    lvl = pred_data["traffic_level"]
    ms = pred_data["inference_ms"]
    print(f"      Predicted Volume: {vol:.1f} veh/hr | Level: {lvl} | Latency: {ms}ms")
    assert vol > 0

    # 3. Model Compare
    print("\n[3/8] Testing /predict/compare (All 3 Models) ...")
    r = client.post("/predict/compare", json=payload)
    assert r.status_code == 200, f"Compare failed: {r.text}"
    comp_data = r.json()
    print(f"      Comparison returned {len(comp_data['results'])} models:")
    for item in comp_data["results"]:
        print(f"        • {item['model']:<18}: {item['predicted_traffic_volume']:>6.0f} veh/hr ({item['traffic_level']})")
    assert len(comp_data["results"]) == 3

    # 4. Analytics
    print("\n[4/8] Testing /analytics (Firestore / Telemetry Aggregation) ...")
    r = client.get("/analytics")
    assert r.status_code == 200, f"Analytics failed: {r.text}"
    an_data = r.json()
    print(f"      Total predictions recorded: {an_data['total_predictions']}")
    print(f"      Average volume: {an_data['avg_volume_veh_hr']} veh/hr | Avg Latency: {an_data['avg_inference_ms']}ms")
    print(f"      Model counts: {an_data['by_model']}")
    assert an_data["total_predictions"] > 0

    # 5. Firebase Storage Status
    print("\n[5/8] Testing /storage/status ...")
    r = client.get("/storage/status")
    assert r.status_code == 200, f"Storage status failed: {r.text}"
    st_data = r.json()
    print(f"      Status: {st_data['status']} | Mode: {st_data['mode']}")
    print(f"      Bucket: {st_data['bucket_name']} | Engine: {st_data['description']}")
    assert st_data["status"] == "connected"

    # 6. Model Sync to Firebase Storage
    print("\n[6/8] Testing /storage/sync-models ...")
    r = client.post("/storage/sync-models")
    assert r.status_code == 200, f"Sync models failed: {r.text}"
    sync_data = r.json()
    print(f"      Synced {sync_data['synced_count']} model artifacts to {sync_data['bucket']}")
    for s in sync_data.get("synced", []):
        size_kb = s["size_bytes"] / 1024
        print(f"        • {s['storage_path']} ({size_kb:.1f} KB)")
    assert sync_data["synced_count"] >= 4

    # 7. Query Stored Artifacts in Firebase Storage
    print("\n[7/8] Testing /storage/models ...")
    r = client.get("/storage/models")
    assert r.status_code == 200, f"List models failed: {r.text}"
    models_list = r.json()
    print(f"      Storage contains {models_list['count']} files under models/ prefix")
    assert models_list["count"] >= 4

    # 8. Export Predictions CSV to Firebase Storage & Download
    print("\n[8/8] Testing /storage/export-predictions and /storage/download ...")
    r = client.post("/storage/export-predictions")
    assert r.status_code == 200, f"Export failed: {r.text}"
    exp_data = r.json()
    storage_info = exp_data["storage"]
    exp_path = storage_info["storage_path"]
    print(f"      Exported {exp_data['exported']} records to: {exp_path}")
    print(f"      Public Firebase URL: {storage_info['public_url']}")

    # Download exported CSV
    dl_url = storage_info["download_url"]
    dl_resp = client.get(dl_url)
    assert dl_resp.status_code == 200, f"Download failed: {dl_resp.status_code}"
    csv_text = dl_resp.text
    assert "predicted_volume" in csv_text, "CSV header missing predicted_volume!"
    print(f"      Download verified: successfully fetched {len(csv_text)} bytes of CSV telemetry data.")

    print("\n" + "=" * 70)
    print("  ALL 8 CLOUD & ML FUNCTIONALITY VERIFICATIONS PASSED 100% PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_system()
