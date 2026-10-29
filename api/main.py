import os
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.logger import logger
from api.schemas import (
    PredictRequest, PredictResponse,
    CompareResponse, CompareItem,
    HealthResponse, ErrorResponse,
)
from api.prediction import load_all_models, get_model_status, predict
from api.firestore_client import log_prediction_to_firestore
from api.storage_client import (
    get_storage_bucket, list_storage_files,
    sync_local_models_to_storage, export_predictions_to_storage,
)

APP_VERSION = "5.0.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development" if os.name == "nt" else "production")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
_START_TIME = time.time()

# ── In-memory session metrics ─────────────────────────────────────────────────
_total_predictions = 0
_per_model = defaultdict(int)
_inference_ms_total = 0.0
_last_predicted_at = None

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"GeoMind API v{APP_VERSION} starting (env={ENVIRONMENT})...")
    status_map = load_all_models()
    for name, ok in status_map.items():
        logger.info(f"  {'OK' if ok else 'FAIL'}: {name}")
    logger.info("GeoMind API ready.")
    yield
    logger.info("GeoMind API shutting down.")


app = FastAPI(
    title="GeoMind — Traffic Forecasting API",
    description=(
        "Classical ML traffic volume prediction from weather + temporal features. "
        "No current traffic count needed. "
        "Deployed on Google Cloud Run + Firebase Hosting + Cloud Firestore."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"project": "GeoMind", "version": APP_VERSION, "docs": "/docs"}


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    model_status = get_model_status()
    all_ok = all(model_status.values()) if model_status else False
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        service="GeoMind",
        environment=ENVIRONMENT,
        models_loaded=model_status,
        api_version=APP_VERSION,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Live session metrics — total calls, per-model counts, avg latency, uptime."""
    n = _total_predictions
    return {
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "total_predictions_served": n,
        "predictions_per_model": dict(_per_model),
        "avg_inference_ms": round(_inference_ms_total / n, 2) if n else 0.0,
        "last_prediction_at": _last_predicted_at,
        "environment": ENVIRONMENT,
        "api_version": APP_VERSION,
    }


# ── Models Info ───────────────────────────────────────────────────────────────
@app.get("/models", tags=["Model Info"])
async def model_info():
    return {
        "available_models": ["xgboost", "random_forest", "ridge_regression"],
        "best_model": "xgboost",
        "dataset": "Metro Interstate Traffic Volume (UCI ML Repository)",
        "target": "traffic volume (vehicles/hr)",
        "infrastructure": {
            "backend": "Google Cloud Run (FastAPI + Uvicorn)",
            "frontend": "Firebase Hosting",
            "database": "Cloud Firestore",
            "storage": "Firebase Storage",
        },
    }


# ── Single Prediction ─────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
@limiter.limit("60/minute")
async def predict_traffic(request: Request, body: PredictRequest):
    """Predict traffic volume for a single hour from weather + temporal inputs."""
    global _total_predictions, _inference_ms_total, _last_predicted_at

    t0 = time.perf_counter()
    obs = body.observation.model_dump()

    try:
        volume, level, congestion_pct, summary = predict(body.model_type, obs)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference error")

    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Update session metrics
    _total_predictions += 1
    _per_model[body.model_type] += 1
    _inference_ms_total += inference_ms
    _last_predicted_at = datetime.now(timezone.utc).isoformat()

    logger.info(f"[{body.model_type}] {volume} veh/hr | {level} | {inference_ms}ms")

    # Log to Firestore (non-blocking)
    try:
        await log_prediction_to_firestore(
            model_name=body.model_type,
            input_datetime=body.observation.date_time,
            volume=volume, level=level,
            congestion_pct=congestion_pct,
            inference_ms=inference_ms, obs=obs,
        )
    except Exception as e:
        logger.warning(f"Firestore log failed (non-fatal): {e}")

    return PredictResponse(
        model_used=body.model_type,
        input_datetime=body.observation.date_time,
        predicted_traffic_volume=volume,
        traffic_level=level,
        congestion_pct=congestion_pct,
        summary=summary,
        inference_ms=inference_ms,
    )


# ── Compare All Models ────────────────────────────────────────────────────────
@app.post("/predict/compare", response_model=CompareResponse, tags=["Prediction"])
@limiter.limit("20/minute")
async def predict_compare(request: Request, body: PredictRequest):
    """Run all 3 models on the same observation — returns a side-by-side comparison."""
    global _total_predictions, _per_model, _inference_ms_total

    obs = body.observation.model_dump()
    results = []

    for model_name in ["xgboost", "random_forest", "ridge_regression"]:
        try:
            t0 = time.perf_counter()
            volume, level, congestion_pct, _ = predict(model_name, obs)
            ms = round((time.perf_counter() - t0) * 1000, 2)
            _total_predictions += 1
            _per_model[model_name] += 1
            _inference_ms_total += ms
            results.append(CompareItem(
                model=model_name,
                predicted_traffic_volume=volume,
                traffic_level=level,
                congestion_pct=congestion_pct,
            ))
        except Exception as e:
            logger.warning(f"Compare: {model_name} failed: {e}")

    return CompareResponse(
        input_datetime=body.observation.date_time,
        weather=body.observation.weather_main,
        results=results,
    )


# ── Analytics (Firestore & Telemetry aggregation) ───────────────────────────
@app.get("/analytics", tags=["Analytics"])
async def get_analytics():
    """
    Aggregate prediction statistics from Cloud Firestore / Telemetry store.
    Powers the Analytics Dashboard in the frontend.
    """
    from api.firestore_client import fetch_prediction_records
    from collections import Counter

    records = fetch_prediction_records(limit=200)
    n = len(records)
    avg_volume = round(sum(r.get("volume", 0) for r in records) / max(n, 1), 1)
    avg_ms     = round(sum(r.get("inference_ms", 0) for r in records) / max(n, 1), 2)
    by_model   = dict(Counter(r.get("model_name", "unknown") for r in records))
    by_level   = dict(Counter(r.get("level", "unknown") for r in records))

    # Hourly average traffic for bar chart (0–23)
    hour_buckets = defaultdict(list)
    for r in records:
        try:
            hour = datetime.fromisoformat(r["input_datetime"]).hour
            hour_buckets[hour].append(r.get("volume", 0))
        except Exception:
            pass
    hourly_avg = {h: round(sum(v) / len(v), 1) for h, v in sorted(hour_buckets.items())}

    # Last 20 predictions for timeline chart
    recent = [
        {"datetime": r.get("input_datetime", ""), "volume": r.get("volume", 0), "level": r.get("level", "")}
        for r in records[:20]
    ]

    return {
        "total_predictions": n,
        "avg_volume_veh_hr": avg_volume,
        "avg_inference_ms": avg_ms,
        "by_model": by_model,
        "by_level": by_level,
        "hourly_avg_volume": hourly_avg,
        "recent": recent,
        "session": {
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "predictions_this_session": _total_predictions,
        },
    }


# ── Firebase Storage ──────────────────────────────────────────────────────────
@app.get("/storage/status", tags=["Firebase Storage"])
async def storage_status():
    from api.storage_client import get_storage_status
    return get_storage_status()


@app.get("/storage/models", tags=["Firebase Storage"])
async def list_models_in_storage():
    from api.storage_client import list_storage_files, get_storage_status
    files = list_storage_files(prefix="models/")
    status_info = get_storage_status()
    return {
        "count": len(files),
        "files": files,
        "bucket": status_info.get("bucket_name"),
        "mode": status_info.get("mode"),
    }


@app.post("/storage/sync-models", tags=["Firebase Storage"])
async def sync_models():
    from api.storage_client import sync_local_models_to_storage
    try:
        return sync_local_models_to_storage()
    except Exception as e:
        logger.error(f"Sync models failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model sync failed: {e}")


@app.post("/storage/export-predictions", tags=["Firebase Storage"])
async def export_predictions():
    from api.firestore_client import fetch_prediction_records
    from api.storage_client import export_predictions_to_storage
    try:
        records = fetch_prediction_records(limit=200)
        result = export_predictions_to_storage(records)
        return {"status": "success", "exported": len(records), "storage": result}
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


@app.get("/storage/download/{file_path:path}", tags=["Firebase Storage"])
async def download_stored_artifact(file_path: str):
    from api.storage_client import get_stored_file_for_download
    target = get_stored_file_for_download(file_path)
    if not target or not target.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found in storage: {file_path}")
    return FileResponse(
        path=str(target),
        filename=target.name,
        media_type="application/octet-stream",
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=False)

