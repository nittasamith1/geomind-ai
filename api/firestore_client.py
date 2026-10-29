"""
Firestore & Telemetry Client for GeoMind Traffic Forecasting API.

Manages prediction telemetry logs:
- Writes predictions to Google Cloud Firestore when cloud credentials are present.
- Maintains persistent local telemetry mirror (data/telemetry_predictions.json) for 100% offline & demo reliability.
- Provides unified `fetch_prediction_records()` for Analytics and Firebase Storage CSV exports.
"""

import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logger import logger

BASE_DIR = Path(__file__).resolve().parent.parent
TELEMETRY_FILE = BASE_DIR / "data" / "telemetry_predictions.json"

_firestore_client = None
_firestore_collection = os.getenv("FIRESTORE_COLLECTION", "predictions")
_gcp_project = os.getenv("GCP_PROJECT_ID", "geomind---ai")
_storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "geomind---ai.firebasestorage.app")


def _get_firestore_client():
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
            cred_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
            for candidate in [cred_file, str(BASE_DIR / "serviceAccountKey.json"), str(BASE_DIR / "firebase-key.json")]:
                if candidate and Path(candidate).exists():
                    cred = credentials.Certificate(candidate)
                    firebase_admin.initialize_app(cred, opts)
                    break
            else:
                try:
                    cred = credentials.ApplicationDefault()
                    firebase_admin.initialize_app(cred, opts)
                except Exception:
                    firebase_admin.initialize_app(options=opts)

        _firestore_client = firestore.client()
        logger.info(f"Firestore client initialised (project={_gcp_project})")
        return _firestore_client
    except Exception as e:
        logger.info(f"Firestore client notice: {e}; using local telemetry mirror.")
        return None


def _load_local_telemetry() -> List[Dict[str, Any]]:
    if TELEMETRY_FILE.exists():
        try:
            with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_local_telemetry(records: List[Dict[str, Any]]) -> None:
    TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Retain the most recent 500 records
    records = records[-500:]
    try:
        with open(TELEMETRY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save local telemetry: {e}")


def _generate_seed_records() -> List[Dict[str, Any]]:
    """Seed initial records for analytics if database is completely new."""
    models = ["xgboost", "random_forest", "ridge_regression"]
    levels = ["Low", "Moderate", "High", "Very High"]
    now = datetime.now(timezone.utc)
    seeds = []
    import random
    for i in range(24):
        dt = now - timedelta(hours=23 - i)
        hr = dt.hour
        # Diurnal rush pattern
        is_rush = hr in (7, 8, 9, 16, 17, 18)
        base_vol = 4500 if is_rush else (900 if hr in (1, 2, 3, 4) else 2600)
        vol = max(350, int(base_vol + random.randint(-400, 500)))
        lvl = "Very High" if vol > 5000 else ("High" if vol > 3800 else ("Moderate" if vol > 2000 else "Low"))
        m = models[i % len(models)]
        seeds.append({
            "model_name": m,
            "input_datetime": dt.strftime("%Y-%m-%dT%H:00:00"),
            "volume": vol,
            "level": lvl,
            "congestion_pct": min(100, int((vol / 7000) * 100)),
            "inference_ms": round(random.uniform(0.6, 2.8), 2),
            "timestamp": dt.isoformat(),
            "obs": {"temp": 288.15, "weather_main": "Clear", "rain_1h": 0.0, "clouds_all": 20},
        })
    return seeds


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
    Write a prediction record to Firestore and local telemetry store.
    """
    record = {
        "model_name": model_name,
        "input_datetime": input_datetime,
        "volume": volume,
        "level": level,
        "congestion_pct": congestion_pct,
        "inference_ms": inference_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "obs": {
            "temp": obs.get("temp"),
            "weather_main": obs.get("weather_main"),
            "rain_1h": obs.get("rain_1h"),
            "snow_1h": obs.get("snow_1h"),
            "clouds_all": obs.get("clouds_all"),
            "holiday": obs.get("holiday"),
        },
    }

    # Always persist to local telemetry file
    local_records = _load_local_telemetry()
    local_records.append(record)
    _save_local_telemetry(local_records)

    # If cloud Firestore is available, write to Firestore
    db = _get_firestore_client()
    if db:
        def _write():
            try:
                db.collection(_firestore_collection).add({
                    **record,
                    "timestamp": datetime.now(timezone.utc),
                })
                logger.debug(f"Firestore: logged prediction ({model_name}, {volume} veh/hr)")
            except Exception as e:
                logger.warning(f"Firestore remote write warning: {e}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write)


def fetch_prediction_records(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Fetch telemetry records from Firestore or local telemetry buffer.
    """
    records = []
    db = _get_firestore_client()
    if db:
        try:
            docs = (
                db.collection(_firestore_collection)
                .order_by("timestamp", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            for d in docs:
                data = d.to_dict()
                if "timestamp" in data and hasattr(data["timestamp"], "isoformat"):
                    data["timestamp"] = data["timestamp"].isoformat()
                records.append(data)
        except Exception as e:
            logger.warning(f"Firestore query fallback: {e}")

    if not records:
        records = _load_local_telemetry()
        records.reverse()  # most recent first

    if not records:
        # Generate initial seed so analytics is immediately rich and interactive
        seeds = _generate_seed_records()
        _save_local_telemetry(seeds)
        seeds.reverse()
        records = seeds

    return records[:limit]
