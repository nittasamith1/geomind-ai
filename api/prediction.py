"""
Inference engine for GeoMind Traffic Forecasting API.

Prediction uses ONLY temporal + weather features.
No current traffic_volume is required at inference time.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.logger import logger
from src.feature_engineering import engineer_features
from src.data_preprocessing import PREPROCESSOR_PATH, get_feature_columns


MODELS_DIR = BASE_DIR / "models" / "ml"
MODEL_PATHS: Dict[str, Path] = {
    "xgboost":          MODELS_DIR / "xgboost.joblib",
    "random_forest":    MODELS_DIR / "random_forest.joblib",
    "ridge_regression": MODELS_DIR / "ridge_regression.joblib",
}

_models: Dict[str, Any] = {}
_pipeline = None
_feature_names: List[str] = []


# Traffic level thresholds (vehicles/hour)
TRAFFIC_LEVELS = [
    (1000,  "Very Low"),
    (2500,  "Low"),
    (4000,  "Moderate"),
    (5500,  "High"),
    (7000,  "Very High"),
    (float("inf"), "Severe"),
]


def _classify_traffic(volume: float) -> str:
    for threshold, label in TRAFFIC_LEVELS:
        if volume < threshold:
            return label
    return "Severe"


def load_all_models() -> Dict[str, bool]:
    global _pipeline, _feature_names
    status = {}

    try:
        data = joblib.load(PREPROCESSOR_PATH)
        _pipeline = data["pipeline"]
        _feature_names = data["feature_names"]
        status["preprocessor"] = True
        logger.info(f"Preprocessor loaded: {len(_feature_names)} features")
    except Exception as e:
        logger.error(f"Preprocessor load failed: {e}")
        status["preprocessor"] = False

    for name, path in MODEL_PATHS.items():
        try:
            if path.exists():
                _models[name] = joblib.load(path)
                status[name] = True
                logger.info(f"Loaded model: {name}")
            else:
                status[name] = False
                logger.warning(f"Model not found: {path}")
        except Exception as e:
            status[name] = False
            logger.error(f"Failed to load {name}: {e}")

    return status


def get_model_status() -> Dict[str, bool]:
    if _pipeline is None:
        load_all_models()
    return {
        "preprocessor": _pipeline is not None,
        **{name: name in _models for name in MODEL_PATHS},
    }


def _build_single_row_df(obs_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row DataFrame from the observation dict.
    No historical traffic_volume needed — just temporal + weather inputs.
    """
    row = {
        "date_time":      pd.to_datetime(obs_dict["date_time"]),
        "temp":           float(obs_dict.get("temp", 288.0)),
        "rain_1h":        float(obs_dict.get("rain_1h", 0.0)),
        "snow_1h":        float(obs_dict.get("snow_1h", 0.0)),
        "clouds_all":     float(obs_dict.get("clouds_all", 0.0)),
        "weather_main":   str(obs_dict.get("weather_main", "Clear")),
        "road_condition": str(obs_dict.get("road_condition", "Dry / Normal")),
        "holiday":        str(obs_dict.get("holiday", "None")),
    }
    return pd.DataFrame([row])


def _generate_summary(volume: float, obs_dict: Dict[str, Any], level: str) -> str:
    dt = pd.to_datetime(obs_dict["date_time"])
    hour = dt.hour
    is_weekend = dt.dayofweek >= 5
    weather = obs_dict.get("weather_main", "Clear")
    road = obs_dict.get("road_condition", "Dry / Normal")

    insights = []
    if 7 <= hour <= 9 and not is_weekend:
        insights.append("Morning commuter rush hour peak.")
    elif 16 <= hour <= 18 and not is_weekend:
        insights.append("Evening return commute surge.")
    elif is_weekend:
        insights.append("Weekend leisurely flow.")
    elif 0 <= hour <= 5:
        insights.append("Overnight low-density highway period.")
    else:
        insights.append("Standard daytime arterial throughput.")

    if weather in ["Snow", "Thunderstorm"] or "Snow" in road or "Ice" in road:
        insights.append("Adverse weather and hazardous surface friction reduce capacity and traffic speed.")
    elif weather in ["Rain", "Drizzle"] or "Wet" in road or "Slippery" in road:
        insights.append("Wet road surface dampens acceleration; exercise standard caution.")
    else:
        insights.append("Clear roadway and optimal driving conditions.")

    return " ".join(insights)


def predict(model_name: str, obs_dict: Dict[str, Any]) -> tuple:
    """
    Predict traffic volume from temporal + weather features only.

    Returns
    -------
    (volume: float, level: str, congestion_pct: int, summary: str)
    """
    if _pipeline is None:
        load_all_models()
    if _pipeline is None:
        raise RuntimeError("Preprocessor not loaded. Run load_all_models() first.")
    if model_name not in _models:
        raise ValueError(
            f"Model '{model_name}' not loaded. Available: {list(_models.keys())}"
        )

    # Build single-row DataFrame
    df = _build_single_row_df(obs_dict)

    # Engineer features (inference mode — no target creation needed)
    df_feat = engineer_features(df, is_training=False)

    if df_feat.empty:
        raise ValueError("Feature engineering returned empty DataFrame.")

    # Transform with fitted preprocessor
    num_cols, cat_cols = get_feature_columns(df_feat)
    X = _pipeline.transform(df_feat[num_cols + cat_cols])

    volume = float(_models[model_name].predict(X)[0])
    volume = max(0.0, round(volume, 1))
    level  = _classify_traffic(volume)
    congestion_pct = min(100, max(0, int(round((volume / 7000.0) * 100))))
    summary = _generate_summary(volume, obs_dict, level)

    return volume, level, congestion_pct, summary

