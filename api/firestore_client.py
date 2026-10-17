"""
Firestore Client for GeoMind Traffic Forecasting API.

Logs each prediction asynchronously to Cloud Firestore for analytics,
monitoring, and model performance tracking over time.

Collection structure:
    predictions/{auto-id}:
        model_name       : str
        input_datetime   : str
        volume           : float
        level            : str
        congestion_pct   : int
        inference_ms     : float
        timestamp        : datetime (server-side)
        obs              : dict  (raw input observation)
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from src.logger import logger

# Firestore is optional — the API works without it if credentials are not configured
_firestore_client = None
_firestore_collection = os.getenv("FIRESTORE_COLLECTION", "predictions")
_gcp_project = os.getenv("GCP_PROJECT_ID", "geomind---ai")
_storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "geomind---ai.firebasestorage.app")


def _get_firestore_client():
    """
    Lazily initialise the Firestore client using Application Default Credentials (ADC).

    On Cloud Run, ADC is automatically available via the service account attached to the
    Cloud Run revision — no service account key file is needed.

    For local development, run:
        gcloud auth application-default login
    """
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            opts = {
                "projectId": _gcp_project,
                "storageBucket": _storage_bucket,
            }
            try:
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, opts)
            except Exception:
                firebase_admin.initialize_app(options=opts)

        _firestore_client = firestore.client()
        logger.info(f"Firestore client initialised (project={_gcp_project or 'default'})")
        return _firestore_client
    except Exception as e:
        logger.warning(f"Firestore client initialisation failed (predictions will not be logged): {e}")
        return None


async def log_prediction_to_firestore(
    model_name: str,
    input_datetime: str,
    volume: float,
    level: str,
    congestion_pct: int,
    inference_ms: float,
    obs: Dict[str, Any],
) -> None:
    """
    Write a prediction record to Firestore asynchronously.

    This is non-blocking — it runs the synchronous Firestore write in a
    thread pool executor so it does not hold up the HTTP response.

    Parameters
    ----------
    model_name      : ML model used (xgboost | random_forest | ridge_regression)
    input_datetime  : ISO 8601 datetime string from the request
    volume          : Predicted traffic volume in vehicles/hr
    level           : Traffic level label (Very Low … Severe)
    congestion_pct  : Congestion percentage (0–100)
    inference_ms    : Model inference latency in milliseconds
    obs             : Raw input observation dict
    """
    db = _get_firestore_client()
    if db is None:
        return  # Firestore not configured — silently skip

    record = {
        "model_name": model_name,
        "input_datetime": input_datetime,
        "volume": volume,
        "level": level,
        "congestion_pct": congestion_pct,
        "inference_ms": inference_ms,
        "timestamp": datetime.now(timezone.utc),
        "obs": {
            "temp": obs.get("temp"),
            "weather_main": obs.get("weather_main"),
            "rain_1h": obs.get("rain_1h"),
            "snow_1h": obs.get("snow_1h"),
            "clouds_all": obs.get("clouds_all"),
            "holiday": obs.get("holiday"),
        },
    }

    def _write():
        db.collection(_firestore_collection).add(record)
        logger.debug(f"Firestore: logged prediction ({model_name}, {volume} veh/hr)")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write)
