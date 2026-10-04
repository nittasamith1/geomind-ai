import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.prediction import load_all_models, predict, get_model_status

def test_pipeline_and_prediction():
    print("Testing model loading...")
    status = load_all_models()
    print("Status:", status)
    assert status.get("preprocessor") is True, "Preprocessor failed to load!"
    assert status.get("xgboost") is True, "XGBoost failed to load!"
    assert status.get("random_forest") is True, "Random Forest failed to load!"
    assert status.get("ridge_regression") is True, "Ridge failed to load!"

    test_obs = {
        "date_time": "2026-10-01T08:00:00",  # Morning rush hour
        "temp": 288.15,                      # ~15 C
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 20.0,
        "weather_main": "Clear",
        "road_condition": "Dry / Normal",
        "holiday": "None",
    }

    print("\nRunning inference across all models:")
    for model_name in ["xgboost", "random_forest", "ridge_regression"]:
        vol, level, pct, summary = predict(model_name, test_obs)
        print(f"  Model: {model_name:18} | Prediction: {vol:6.1f} veh/hr | Level: {level:9} | Cap: {pct}%")
        print(f"         Summary: {summary}")
        assert vol > 0, f"Volume should be > 0, got {vol}"
        assert 0 <= pct <= 100, f"Capacity should be between 0 and 100, got {pct}"
        assert len(summary) > 0, "Summary should not be empty!"

    # Night time observation (2 AM)
    night_obs = {
        "date_time": "2026-10-01T02:00:00",  # Night
        "temp": 283.15,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 0.0,
        "weather_main": "Clear",
        "road_condition": "Dry / Normal",
        "holiday": "None",
    }
    vol_night, level_night, pct_night, _ = predict("xgboost", night_obs)
    vol_rush, _, pct_rush, _ = predict("xgboost", test_obs)
    print(f"\nComparing Rush hour (8 AM: {vol_rush} veh/hr, {pct_rush}%) vs Night (2 AM: {vol_night} veh/hr, {pct_night}%):")
    assert vol_rush > vol_night, "Rush hour traffic should be higher than 2 AM night traffic!"
    print("  Comparison validation passed: rush hour traffic is significantly higher than night traffic!")

    # Bad weather test (Heavy rain & storm during rush hour with flooded road)
    storm_obs = {
        "date_time": "2026-10-01T08:00:00",
        "temp": 280.15,
        "rain_1h": 25.0,
        "snow_1h": 0.0,
        "clouds_all": 100.0,
        "weather_main": "Thunderstorm",
        "road_condition": "Flooded / Standing Water",
        "holiday": "None",
    }
    vol_storm, level_storm, pct_storm, summary_storm = predict("xgboost", storm_obs)
    print(f"Storm weather traffic prediction at 8 AM: {vol_storm} veh/hr ({level_storm}, {pct_storm}%)")
    print(f"  Storm summary: {summary_storm}")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline_and_prediction()
