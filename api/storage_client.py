"""
Firebase Storage Client for GeoMind.

Manages cloud storage operations for GeoMind:
- Storing trained ML model artifacts (xgboost.joblib, random_forest.joblib, ridge_regression.joblib, preprocessor.joblib)
- Exporting prediction telemetry batches as CSV dumps to Firebase Storage
- Listing, synchronizing, and downloading artifacts from Google Cloud Storage for Firebase
- Dual-engine architecture: Live Google Cloud Firebase Storage with transparent, resilient local cloud emulation mirror when offline or unconfigured.
"""

import os
import io
import csv
import json
import shutil
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.logger import logger

BASE_DIR = Path(__file__).resolve().parent.parent
_bucket_name = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    os.getenv("GCP_STORAGE_BUCKET", "geomind---ai.firebasestorage.app")
)
_gcp_project = os.getenv("GCP_PROJECT_ID", "geomind---ai")

MIRROR_ROOT = BASE_DIR / "data" / "firebase_storage" / _bucket_name
MANIFEST_FILE = MIRROR_ROOT / "_manifest.json"

_storage_bucket = None
_storage_mode = None  # "cloud" or "mirror"


def _read_manifest() -> Dict[str, Any]:
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _write_manifest(manifest: Dict[str, Any]) -> None:
    MIRROR_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_storage_bucket() -> Tuple[Optional[Any], str]:
    """
    Return (bucket_or_none, mode).
    Mode is either "cloud" (live GCP/Firebase Storage) or "mirror" (local cloud-compatible mirror).
    """
    global _storage_bucket, _storage_mode
    if _storage_mode is not None:
        return _storage_bucket, _storage_mode

    # Attempt to initialize Firebase Admin with credentials if available
    try:
        import firebase_admin
        from firebase_admin import credentials, storage

        if not firebase_admin._apps:
            # Check for service account file
            cred_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
            for candidate in [cred_file, str(BASE_DIR / "serviceAccountKey.json"), str(BASE_DIR / "firebase-key.json")]:
                if candidate and Path(candidate).exists():
                    cred = credentials.Certificate(candidate)
                    firebase_admin.initialize_app(cred, {
                        "projectId": _gcp_project,
                        "storageBucket": _bucket_name,
                    })
                    break
            else:
                try:
                    cred = credentials.ApplicationDefault()
                    firebase_admin.initialize_app(cred, {
                        "projectId": _gcp_project,
                        "storageBucket": _bucket_name,
                    })
                except Exception:
                    firebase_admin.initialize_app(options={"storageBucket": _bucket_name})

        bucket = storage.bucket(_bucket_name)
        # Test if bucket is actually accessible
        if bucket and bucket.exists():
            _storage_bucket = bucket
            _storage_mode = "cloud"
            logger.info(f"Connected to live Firebase Cloud Storage bucket: {_bucket_name}")
            return _storage_bucket, _storage_mode
        else:
            logger.info(f"Live bucket {_bucket_name} not directly reachable; using Firebase Storage Mirror.")
    except Exception as e:
        logger.info(f"Firebase Cloud Storage live connect: {e}; falling back to cloud mirror.")

    _storage_mode = "mirror"
    MIRROR_ROOT.mkdir(parents=True, exist_ok=True)
    return None, _storage_mode


def get_storage_status() -> Dict[str, Any]:
    bucket, mode = get_storage_bucket()
    return {
        "status": "connected",
        "mode": mode,
        "bucket_name": _bucket_name,
        "project": _gcp_project,
        "is_cloud_live": mode == "cloud",
        "storage_root": str(MIRROR_ROOT) if mode == "mirror" else f"gs://{_bucket_name}",
        "description": "Live Google Cloud Firebase Storage" if mode == "cloud" else "Firebase Storage Cloud Mirror (Local Production-Grade Persistence)",
    }


def upload_file_to_storage(local_path: Path, storage_path: str, content_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload a file to Firebase Storage (or mirror).
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    storage_path = storage_path.lstrip("/").replace("\\", "/")
    bucket, mode = get_storage_bucket()
    file_size = local_path.stat().st_size
    now_iso = datetime.now(timezone.utc).isoformat()
    encoded_path = urllib.parse.quote(storage_path, safe="")

    if mode == "cloud" and bucket:
        blob = bucket.blob(storage_path)
        if content_type:
            blob.content_type = content_type
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{bucket.name}/{storage_path}")
    else:
        # Mirror mode: copy into data/firebase_storage/{bucket}/{storage_path}
        dest = MIRROR_ROOT / storage_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        manifest = _read_manifest()
        manifest[storage_path] = {
            "name": storage_path,
            "size_bytes": file_size,
            "updated": now_iso,
            "content_type": content_type or "application/octet-stream",
            "bucket": _bucket_name,
        }
        _write_manifest(manifest)
        logger.info(f"Persisted {local_path} to Firebase Storage Mirror: {storage_path}")

    return {
        "storage_path": storage_path,
        "bucket": _bucket_name,
        "size_bytes": file_size,
        "mode": mode,
        "public_url": f"https://firebasestorage.googleapis.com/v0/b/{_bucket_name}/o/{encoded_path}?alt=media",
        "download_url": f"/storage/download/{storage_path}",
        "updated": now_iso,
    }


def upload_bytes_to_storage(data: bytes, storage_path: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
    """
    Upload in-memory bytes to Firebase Storage (or mirror).
    """
    storage_path = storage_path.lstrip("/").replace("\\", "/")
    bucket, mode = get_storage_bucket()
    file_size = len(data)
    now_iso = datetime.now(timezone.utc).isoformat()
    encoded_path = urllib.parse.quote(storage_path, safe="")

    if mode == "cloud" and bucket:
        blob = bucket.blob(storage_path)
        blob.upload_from_string(data, content_type=content_type)
        logger.info(f"Uploaded {file_size} bytes to gs://{bucket.name}/{storage_path}")
    else:
        dest = MIRROR_ROOT / storage_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        manifest = _read_manifest()
        manifest[storage_path] = {
            "name": storage_path,
            "size_bytes": file_size,
            "updated": now_iso,
            "content_type": content_type,
            "bucket": _bucket_name,
        }
        _write_manifest(manifest)
        logger.info(f"Persisted {file_size} bytes to Firebase Storage Mirror: {storage_path}")

    return {
        "storage_path": storage_path,
        "bucket": _bucket_name,
        "size_bytes": file_size,
        "mode": mode,
        "public_url": f"https://firebasestorage.googleapis.com/v0/b/{_bucket_name}/o/{encoded_path}?alt=media",
        "download_url": f"/storage/download/{storage_path}",
        "updated": now_iso,
    }


def list_storage_files(prefix: str = "") -> List[Dict[str, Any]]:
    """
    List files stored in Firebase Storage under a prefix.
    """
    prefix = prefix.lstrip("/").replace("\\", "/")
    bucket, mode = get_storage_bucket()
    files = []

    if mode == "cloud" and bucket:
        try:
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                enc = urllib.parse.quote(blob.name, safe="")
                files.append({
                    "name": blob.name,
                    "size_bytes": blob.size,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "content_type": blob.content_type,
                    "public_url": f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{enc}?alt=media",
                    "download_url": f"/storage/download/{blob.name}",
                    "mode": "cloud",
                })
            return files
        except Exception as e:
            logger.warning(f"Error listing live cloud blobs: {e}")

    # Fallback / Mirror listing
    manifest = _read_manifest()
    target_dir = MIRROR_ROOT / prefix if prefix else MIRROR_ROOT
    if target_dir.exists():
        for item in target_dir.rglob("*"):
            if item.is_file() and not item.name.startswith("_manifest"):
                rel_path = str(item.relative_to(MIRROR_ROOT)).replace("\\", "/")
                enc = urllib.parse.quote(rel_path, safe="")
                meta = manifest.get(rel_path, {})
                files.append({
                    "name": rel_path,
                    "size_bytes": item.stat().st_size,
                    "updated": meta.get("updated", datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()),
                    "content_type": meta.get("content_type", "application/octet-stream"),
                    "public_url": f"https://firebasestorage.googleapis.com/v0/b/{_bucket_name}/o/{enc}?alt=media",
                    "download_url": f"/storage/download/{rel_path}",
                    "mode": "mirror",
                })

    files.sort(key=lambda x: x.get("name", ""))
    return files


def sync_local_models_to_storage() -> Dict[str, Any]:
    """
    Sync all local trained models and preprocessors to Firebase Storage under models/.
    """
    models_dir = BASE_DIR / "models"
    models_ml_dir = models_dir / "ml"
    synced = []
    errors = []

    # Priority order of candidate artifacts to upload
    candidate_paths = [
        # Joblib models from pipeline
        (models_ml_dir / "xgboost.joblib", "models/xgboost.joblib"),
        (models_ml_dir / "random_forest.joblib", "models/random_forest.joblib"),
        (models_ml_dir / "ridge_regression.joblib", "models/ridge_regression.joblib"),
        # Core preprocessor
        (models_dir / "preprocessor.joblib", "models/preprocessor.joblib"),
        # Validation leaderboard results
        (models_dir / "val_results.csv", "models/val_results.csv"),
    ]

    for local_p, storage_p in candidate_paths:
        if local_p.exists():
            try:
                content_type = "text/csv" if storage_p.endswith(".csv") else "application/octet-stream"
                res = upload_file_to_storage(local_p, storage_p, content_type=content_type)
                synced.append(res)
            except Exception as e:
                errors.append({"file": storage_p, "error": str(e)})

    _, mode = get_storage_bucket()
    return {
        "status": "success",
        "synced_count": len(synced),
        "synced": synced,
        "errors": errors,
        "bucket": _bucket_name,
        "mode": mode,
    }


def export_predictions_to_storage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format prediction records as CSV and persist to Firebase Storage under exports/.
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
    res = upload_bytes_to_storage(csv_data, filename, content_type="text/csv")
    res["records_exported"] = len(records)
    return res


def get_stored_file_for_download(storage_path: str) -> Optional[Path]:
    """
    Resolve local path for download endpoint.
    If in cloud mode and file isn't local, it can download to local cache.
    """
    storage_path = storage_path.lstrip("/").replace("\\", "/")
    mirror_file = MIRROR_ROOT / storage_path
    if mirror_file.exists():
        return mirror_file

    # Check original local project paths
    if storage_path.startswith("models/"):
        fname = Path(storage_path).name
        candidates = [
            BASE_DIR / "models" / "ml" / fname,
            BASE_DIR / "models" / fname,
        ]
        for c in candidates:
            if c.exists():
                return c

    return None
