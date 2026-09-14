"""
GeoMind AI - FastAPI Prediction Engine
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Model lifecycle management:
1. Lazy-loads all trained artifacts on first request (ML + DL).
2. Runs feature engineering + preprocessing on raw observation input.
3. Supports XGBoost, Random Forest (ML) and LSTM, GRU (PyTorch DL).
4. Returns predicted next-hour traffic volume (vehicles/hr).
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException
from src.feature_engineering import engineer_features
from src.data_preprocessing import TrafficPreprocessor

# ──────────────────────────────────────────────────────────────────────────────
# Model Registry — all artifact paths
# ──────────────────────────────────────────────────────────────────────────────
ML_MODEL_PATHS: Dict[str, Path] = {
    "xgboost":       Path("models/ml/xgboost.joblib"),
    "random_forest": Path("models/ml/random_forest.joblib"),
}

DL_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "lstm": {"path": Path("models/dl/lstm_context_l_6.pt"),  "seq_len": 6,  "arch": "LSTM"},
    "gru":  {"path": Path("models/dl/gru_context_l_12.pt"),  "seq_len": 12, "arch": "GRU"},
}

PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

# ──────────────────────────────────────────────────────────────────────────────
# Global in-memory model store (lazy-loaded singletons)
# ──────────────────────────────────────────────────────────────────────────────
_ml_models:    Dict[str, Any]  = {}
_dl_models:    Dict[str, Any]  = {}
_preprocessor: Optional[TrafficPreprocessor] = None
_feature_names: List[str] = []


def load_all_models() -> Dict[str, bool]:
    """
    Pre-loads all available model artifacts into memory.
    Called at API startup for zero-latency first requests.
    Returns a dict of {model_name: load_success}.
    """
    global _preprocessor, _feature_names
    status: Dict[str, bool] = {}

    # Load preprocessor
    try:
        _preprocessor = TrafficPreprocessor()
        _preprocessor.load_artifact()
        _feature_names = _preprocessor.feature_names
        logger.info(f"Preprocessor loaded: {len(_feature_names)} features")
        status["preprocessor"] = True
    except Exception as e:
        logger.error(f"Failed to load preprocessor: {e}")
        status["preprocessor"] = False

    # Load ML models
    for name, path in ML_MODEL_PATHS.items():
        try:
            if path.exists():
                _ml_models[name] = joblib.load(path)
                logger.info(f"ML model '{name}' loaded from {path}")
                status[name] = True
            else:
                logger.warning(f"ML model artifact not found: {path}")
                status[name] = False
        except Exception as e:
            logger.error(f"Failed to load ML model '{name}': {e}")
            status[name] = False

    # Load DL models
    for name, cfg in DL_MODEL_CONFIGS.items():
        try:
            import torch
            from src.train_dl import TrafficLSTM, TrafficGRU

            ckpt_path = cfg["path"]
            if not ckpt_path.exists():
                logger.warning(f"DL model artifact not found: {ckpt_path}")
                status[name] = False
                continue

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            input_dim = len(_feature_names) if _feature_names else 39
            arch = cfg["arch"]

            if arch == "LSTM":
                net = TrafficLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
            else:
                net = TrafficGRU(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)

            net.load_state_dict(checkpoint["model_state_dict"])
            net.eval()

            _dl_models[name] = {
                "net": net,
                "seq_len": checkpoint.get("seq_len", cfg["seq_len"]),
                "target_scaler_mean": checkpoint.get("target_scaler_mean", 0.0),
                "target_scaler_scale": checkpoint.get("target_scaler_scale", 1.0),
            }
            logger.info(f"DL model '{name}' loaded (seq_len={cfg['seq_len']})")
            status[name] = True

        except Exception as e:
            logger.error(f"Failed to load DL model '{name}': {e}")
            status[name] = False

    return status


def _build_padded_dataframe(
    observations: List[Dict[str, Any]],
    n_preceding: int = 30
) -> pd.DataFrame:
    """
    Constructs a continuous time-series DataFrame from input observations,
    padding synthetic predecessors before the first observation and a dummy
    successor after the last observation.

    This ensures:
    1. Autoregressive lags (up to 24h) and rolling windows (up to 24h) are fully
       computed without dropping real observations.
    2. The target variable (t+1) is computable for all real observations, preventing
       engineer_features() from dropping the final real observation.
    3. All date_time values are native pd.Timestamp, avoiding any string format errors.
    """
    first_obs = observations[0]
    first_dt = pd.to_datetime(first_obs["date_time"])

    rows = []
    # 1. Synthetic predecessors (stepping backwards 1 hour at a time)
    for k in range(n_preceding, 0, -1):
        synth = dict(first_obs)
        synth["date_time"] = first_dt - pd.Timedelta(hours=k)
        synth["future_traffic_volume"] = 0.0
        rows.append(synth)

    # 2. Actual observations
    for obs in observations:
        actual = dict(obs)
        actual["date_time"] = pd.to_datetime(obs["date_time"])
        actual["future_traffic_volume"] = 0.0
        rows.append(actual)

    # 3. Synthetic successor row at the end so create_target_variable doesn't produce NaN for the last actual observation
    last_obs = observations[-1]
    last_dt = pd.to_datetime(last_obs["date_time"])
    succ = dict(last_obs)
    succ["date_time"] = last_dt + pd.Timedelta(hours=1)
    succ["future_traffic_volume"] = 0.0
    rows.append(succ)

    df = pd.DataFrame(rows)
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df


def predict_ml(model_name: str, observation_dict: Dict[str, Any]) -> float:
    """
    Runs inference with a traditional ML model (XGBoost or Random Forest).

    Builds a padded context window with 30 synthetic predecessors so that
    autoregressive lags (1h, 2h, 3h, 24h) and rolling window features are computed.
    Only the feature vector for the actual observation (last row) is used.

    Args:
        model_name: One of 'xgboost', 'random_forest'
        observation_dict: Raw observation payload dict

    Returns:
        Predicted traffic volume (vehicles/hr) as float
    """
    global _preprocessor, _ml_models

    if _preprocessor is None:
        raise RuntimeError("Preprocessor not loaded. Call load_all_models() at startup.")
    if model_name not in _ml_models:
        raise ValueError(f"ML model '{model_name}' not available. Loaded: {list(_ml_models.keys())}")

    # Build padded sequence (30 synthetic predecessors + actual observation + 1 successor)
    df_padded = _build_padded_dataframe([observation_dict], n_preceding=30)
    df_feat = engineer_features(df_padded)

    if df_feat.empty:
        raise ValueError("Feature engineering produced empty DataFrame. Check input observation validity.")

    # Use the last row (the actual observation's features)
    df_last = df_feat.iloc[[-1]]
    X, _ = _preprocessor.transform(df_last)
    model = _ml_models[model_name]
    prediction = float(model.predict(X)[0])
    return max(0.0, round(prediction, 2))


def predict_dl(model_name: str, observations: List[Dict[str, Any]]) -> float:
    """
    Runs inference with a PyTorch DL model (LSTM or GRU).

    Automatically pads short observation sequences with synthetic predecessors
    so that both feature engineering and sequential models have sufficient history.

    Args:
        model_name: One of 'lstm', 'gru'
        observations: List of sequential hourly observations (oldest → newest)

    Returns:
        Predicted traffic volume (vehicles/hr) as float
    """
    global _preprocessor, _dl_models

    if _preprocessor is None:
        raise RuntimeError("Preprocessor not loaded. Call load_all_models() at startup.")
    if model_name not in _dl_models:
        raise ValueError(f"DL model '{model_name}' not available. Loaded: {list(_dl_models.keys())}")

    if not observations:
        raise ValueError("Cannot predict on empty observations list.")

    import torch
    cfg = _dl_models[model_name]
    seq_len = cfg["seq_len"]
    target_mean = cfg["target_scaler_mean"]
    target_scale = cfg["target_scaler_scale"]

    # Build padded dataframe with enough history for 24h lags + seq_len
    n_preceding = max(30, seq_len + 5)
    df_padded = _build_padded_dataframe(observations, n_preceding=n_preceding)
    df_feat = engineer_features(df_padded)

    if len(df_feat) < seq_len:
        raise ValueError(f"Engineered features yielded {len(df_feat)} rows, which is less than seq_len={seq_len}.")

    X, _ = _preprocessor.transform(df_feat)
    # Take the last seq_len rows
    X_seq = X[-seq_len:]
    X_tensor = torch.tensor(X_seq[np.newaxis, :, :], dtype=torch.float32)  # (1, seq_len, features)

    net = cfg["net"]
    with torch.no_grad():
        pred_scaled = net(X_tensor).item()

    # Inverse target standardization: unscaled = scaled * scale + mean
    pred_unscaled = pred_scaled * target_scale + target_mean
    return max(0.0, round(float(pred_unscaled), 2))


def get_loaded_model_status() -> Dict[str, bool]:
    """Returns current load status of all models."""
    status = {
        "preprocessor": _preprocessor is not None,
    }
    for name in ML_MODEL_PATHS:
        status[name] = name in _ml_models
    for name in DL_MODEL_CONFIGS:
        status[name] = name in _dl_models
    return status
