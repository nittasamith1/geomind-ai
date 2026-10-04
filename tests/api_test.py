r"""
Quick API test for GeoMind predictions.
Run: .venv\Scripts\python.exe tests/api_test.py
"""
import urllib.request
import json

API = "http://127.0.0.1:8000"


def post(endpoint, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{API}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def get(endpoint):
    with urllib.request.urlopen(f"{API}{endpoint}", timeout=10) as resp:
        return json.loads(resp.read())


def make_obs(dt, temp_c=20, rain=0, snow=0, clouds=10, weather="Clear",
             road="Dry / Normal", holiday="None"):
    return {
        "observation": {
            "date_time": dt,
            "temp": round(temp_c + 273.15, 2),
            "rain_1h": rain,
            "snow_1h": snow,
            "clouds_all": clouds,
            "weather_main": weather,
            "road_condition": road,
            "holiday": holiday,
        },
        "model_type": "xgboost",
    }


def run_tests():
    print("=" * 65)
    print("  GeoMind API Prediction Validation")
    print("=" * 65)

    # Health check
    h = get("/health")
    print(f"\nHealth: {h.get('status')}  |  models={h.get('models_loaded')}")

    scenarios = [
        ("Weekday Rush (Tue 8am Clear)",
         make_obs("2024-10-01T08:00:00", temp_c=20, clouds=10)),
        ("Weekday Night (Tue 3am Clear)",
         make_obs("2024-10-01T03:00:00", temp_c=15, clouds=0)),
        ("Weekday Midday (Tue 12pm Clear)",
         make_obs("2024-10-01T12:00:00", temp_c=22, clouds=20)),
        ("Weekday Evening Rush (Tue 5pm Clear)",
         make_obs("2024-10-01T17:00:00", temp_c=20, clouds=20)),
        ("Weekend Morning (Sat 9am Clear)",
         make_obs("2024-10-05T09:00:00", temp_c=20, clouds=15)),
        ("Christmas Morning (9am Clear)",
         make_obs("2024-12-25T09:00:00", temp_c=5, clouds=30, holiday="Christmas Day")),
        ("Snowstorm Rush (Mon 8am Snow)",
         make_obs("2024-02-19T08:00:00", temp_c=-5, snow=5, clouds=90,
                  weather="Snow", road="Snow / Ice Covered")),
        ("Thunderstorm Rush (Wed 8am Storm)",
         make_obs("2024-10-02T08:00:00", temp_c=18, rain=10, clouds=95,
                  weather="Thunderstorm", road="Wet / Damp")),
        ("Summer Weekend (Sat 2pm Clear)",
         make_obs("2024-07-06T14:00:00", temp_c=30, clouds=5)),
        ("Winter Night (Mon 2am Snow)",
         make_obs("2024-01-15T02:00:00", temp_c=-10, snow=2, clouds=80,
                  weather="Snow", road="Snow / Ice Covered")),
    ]

    print(f"\n{'Scenario':<40} {'Volume':>8} {'Level':<12}")
    print("-" * 65)

    for name, payload in scenarios:
        try:
            resp = post("/predict", payload)
            vol = resp.get("predicted_traffic_volume", "ERR")
            lvl = resp.get("traffic_level", "ERR")
            print(f"{name:<40} {str(vol):>8} {lvl:<12}")
        except Exception as e:
            print(f"{name:<40} ERROR: {e}")

    print("\n" + "=" * 65)
    print("  All 3 models — Rush Hour Comparison")
    print("=" * 65)
    rush_obs_base = make_obs("2024-10-01T08:00:00", temp_c=20, clouds=10)
    for model in ["xgboost", "random_forest", "ridge_regression"]:
        payload = {**rush_obs_base, "model_type": model}
        try:
            resp = post("/predict", payload)
            vol = resp.get("predicted_traffic_volume")
            lvl = resp.get("traffic_level")
            ms = resp.get("inference_ms")
            print(f"  {model:<20} {vol:>8.0f} veh/hr  {lvl:<12}  ({ms} ms)")
        except Exception as e:
            print(f"  {model:<20} ERROR: {e}")

    print("=" * 65)


if __name__ == "__main__":
    run_tests()
