"""
Firebase Storage Client for GeoMind.

Manages cloud storage operations for GeoMind:
- Uploading & storing trained ML model artifacts (xgboost.pkl, random_forest.pkl, ridge_regression.pkl, preprocessor.joblib)
- Uploading prediction export reports and telemetry CSV dumps
- Fetching model artifacts or metadata from Cloud Storage for Firebase
"""

import os
import io
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.logger import logger

_storage_bucket = None
_bucket_name = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    os.getenv("GCP_STORAGE_BUCKET", "geomind---ai.firebasestorage.app")
)
_gcp_project = os.getenv("GCP_PROJECT_ID", "geomind---ai")


def get_storage_bucket():
    """
    Lazily initialize and return the Firebase Storage bucket.
    Uses Application Default Credentials (ADC) or initialized firebase_admin app.
    """
    global _storage_bucket
    if _storage_bucket is not None:
        return _storage_bucket

    try:
        import firebase_admin
        from firebase_admin import credentials, storage

        if not firebase_admin._apps:
            if _gcp_project:
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, {
                    "projectId": _gcp_project,
                    "storageBucket": _bucket_name,
                })
            else:
                firebase_admin.initialize_app(options={"storageBucket": _bucket_name})

        _storage_bucket = storage.bucket(_bucket_name)
        logger.info(f"Firebase Storage bucket initialized: {_bucket_name}")
        return _storage_bucket
    except Exception as e:
        logger.warning(f"Firebase Storage initialization notice: {e}")
        return None


def upload_file_to_storage(local_path: Path, storage_path: str, content_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload a local file to Firebase Storage.
    """
    bucket = get_storage_bucket()
    if not bucket:
        raise RuntimeError("Firebase Storage bucket is not available or not configured.")

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    blob = bucket.blob(storage_path)
    if content_type:
        blob.content_type = content_type

    blob.upload_from_filename(str(local_path))
    logger.info(f"Uploaded {local_path} to gs://{bucket.name}/{storage_path}")

    return {
        "storage_path": storage_path,
        "bucket": bucket.name,
        "size_bytes": local_path.stat().st_size,
        "public_url": f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{storage_path.replace('/', '%2F')}?alt=media",
    }


def upload_bytes_to_storage(data: bytes, storage_path: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
    """
    Upload in-memory bytes to Firebase Storage.
    """
    bucket = get_storage_bucket()
    if not bucket:
        raise RuntimeError("Firebase Storage bucket is not available or not configured.")

    blob = bucket.blob(storage_path)
    blob.upload_from_string(data, content_type=content_type)
    logger.info(f"Uploaded {len(data)} bytes to gs://{bucket.name}/{storage_path}")

    return {
        "storage_path": storage_path,
        "bucket": bucket.name,
        "size_bytes": len(data),
        "public_url": f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{storage_path.replace('/', '%2F')}?alt=media",
    }


def list_storage_files(prefix: str = "") -> List[Dict[str, Any]]:
    """
    List files stored in Firebase Storage under a prefix (e.g. 'models/' or 'exports/').
    """
    bucket = get_storage_bucket()
    if not bucket:
        return []

    try:
        blobs = bucket.list_blobs(prefix=prefix)
        files = []
        for blob in blobs:
            files.append({
                "name": blob.name,
                "size_bytes": blob.size,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "content_type": blob.content_type,
                "public_url": f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media",
            })
        return files
    except Exception as e:
        logger.warning(f"Failed to list Firebase Storage files: {e}")
        return []


def sync_local_models_to_storage() -> Dict[str, Any]:
    """
    Sync all local trained models and preprocessors in models/ to Firebase Storage under models/.
    """
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    synced = []
    errors = []

    model_files = [
        "xgboost.pkl",
        "random_forest.pkl",
        "ridge_regression.pkl",
        "preprocessor.joblib",
        "val_results.csv",
    ]

    for fname in model_files:
        p = models_dir / fname
        if p.exists():
            try:
                res = upload_file_to_storage(p, f"models/{fname}")
                synced.append(res)
            except Exception as e:
                errors.append({"file": fname, "error": str(e)})

    return {
        "synced_count": len(synced),
        "synced": synced,
        "errors": errors,
        "bucket": _bucket_name,
    }


def export_predictions_to_storage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of prediction records, format as CSV and save to Firebase Storage under exports/.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "model_name", "input_datetime",
        "predicted_volume", "traffic_level", "congestion_pct", "inference_ms"
    ])

    for r in records:
        writer.writerow([
            r.get("timestamp", datetime.now(timezone.utc).isoformat()),
            r.get("model_name", ""),
            r.get("input_datetime", ""),
            r.get("volume", 0),
            r.get("level", ""),
            r.get("congestion_pct", 0),
            r.get("inference_ms", 0),
        ])

    csv_data = output.getvalue().encode("utf-8")
    filename = f"exports/predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return upload_bytes_to_storage(csv_data, filename, content_type="text/csv")
