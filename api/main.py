import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.logger import logger
from api.schemas import PredictRequest, PredictResponse, HealthResponse, ErrorResponse
from api.prediction import load_all_models, get_model_status, predict
from api.firestore_client import log_prediction_to_firestore
from api.storage_client import (
    get_storage_bucket,
    list_storage_files,
    sync_local_models_to_storage,
    export_predictions_to_storage,
)

# Cloud Run passes PORT via environment variable (default 8080 in Docker/Cloud Run, 8000 locally)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development" if os.name == "nt" else "production")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"GeoMind API starting (env={ENVIRONMENT}) — loading models...")
    status_map = load_all_models()
    for name, ok in status_map.items():
        logger.info(f"  {'OK' if ok else 'FAIL'}: {name}")
    logger.info("GeoMind API ready for traffic forecasting inference.")
    yield
    logger.info("GeoMind API shutting down.")


app = FastAPI(
    title="GeoMind — Traffic Forecasting API",
    description=(
        "Classical ML traffic volume prediction from weather + temporal features. "
        "No current traffic count needed — just date/time, weather, and holiday. "
        "Deployed on Google Cloud Run with Firebase Hosting frontend & Firebase Storage."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

# CORS Configuration — allows localhost ports (8000, 8080, 5500), file:// and Firebase Hosting
cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
if cors_origins_env:
    allow_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend assets (for local development)
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", tags=["Root"])
async def root():
    frontend_index = frontend_dir / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {
        "project": "GeoMind — Urban Traffic Forecasting",
        "version": "4.0.0",
        "environment": ENVIRONMENT,
        "hosting": "Firebase Hosting + Google Cloud Run",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    model_status = get_model_status()
    all_ok = all(model_status.values()) if model_status else False
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        service="GeoMind",
        environment=ENVIRONMENT,
        models_loaded=model_status,
        api_version="4.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_traffic(request: PredictRequest):
    t0 = time.perf_counter()
    obs = request.observation.model_dump()

    try:
        volume, level, congestion_pct, summary = predict(request.model_type, obs)
    except ValueError as e:
        logger.warning(f"Prediction validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Prediction service unavailable: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction inference error"
        )

    inference_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"Prediction [{request.model_type}]: {volume} veh/hr "
        f"({level}, {congestion_pct}% cap) in {inference_ms}ms"
    )

    # Log prediction asynchronously to Firestore (non-blocking, errors are swallowed)
    try:
        await log_prediction_to_firestore(
            model_name=request.model_type,
            input_datetime=request.observation.date_time,
            volume=volume,
            level=level,
            congestion_pct=congestion_pct,
            inference_ms=inference_ms,
            obs=obs,
        )
    except Exception as fs_err:
        logger.warning(f"Firestore logging failed (non-fatal): {fs_err}")

    return PredictResponse(
        model_used=request.model_type,
        input_datetime=request.observation.date_time,
        predicted_traffic_volume=volume,
        traffic_level=level,
        congestion_pct=congestion_pct,
        summary=summary,
        inference_ms=inference_ms,
    )


@app.get("/models", tags=["Model Info"])
async def model_info():
    return {
        "available_models": ["xgboost", "random_forest", "ridge_regression"],
        "best_model": "xgboost",
        "dataset": "Metro Interstate Traffic Volume (UCI ML Repository)",
        "target": "traffic volume (vehicles/hr)",
        "features": "temporal + weather features (no current traffic count needed)",
        "infrastructure": {
            "backend": "Google Cloud Run (FastAPI + Uvicorn)",
            "frontend": "Firebase Hosting",
            "database": "Cloud Firestore (prediction logs)",
            "storage": "Firebase Storage (models & reports bucket)",
        },
        "inputs_required": [
            "date_time", "temp", "weather_main",
            "rain_1h", "snow_1h", "clouds_all", "holiday"
        ],
    }


@app.get("/storage/status", tags=["Firebase Storage"])
async def storage_status():
    """Check Firebase Storage bucket connectivity and configuration."""
    bucket = get_storage_bucket()
    return {
        "status": "connected" if bucket else "unconfigured",
        "bucket_name": bucket.name if bucket else os.getenv("FIREBASE_STORAGE_BUCKET", "geomind---ai.firebasestorage.app"),
        "project": os.getenv("GCP_PROJECT_ID", "geomind---ai"),
    }


@app.get("/storage/models", tags=["Firebase Storage"])
async def list_models_in_storage():
    """List model artifacts persisted in Firebase Storage."""
    files = list_storage_files(prefix="models/")
    return {
        "count": len(files),
        "files": files,
    }


@app.post("/storage/sync-models", tags=["Firebase Storage"])
async def sync_models():
    """Upload local trained models and preprocessor to Firebase Storage."""
    try:
        res = sync_local_models_to_storage()
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model sync to Firebase Storage failed: {e}",
        )


@app.post("/storage/export-predictions", tags=["Firebase Storage"])
async def export_predictions():
    """Export recent Firestore prediction records to a CSV file in Firebase Storage."""
    try:
        from api.firestore_client import _get_firestore_client
        client = _get_firestore_client()
        records = []
        if client:
            docs = client.collection("predictions").order_by("timestamp", direction="DESCENDING").limit(100).stream()
            for d in docs:
                data = d.to_dict()
                records.append(data)

        if not records:
            # Sample fallback record if no predictions logged yet
            records.append({
                "model_name": "xgboost",
                "input_datetime": "2026-10-05T12:00:00",
                "volume": 3200,
                "level": "Moderate",
                "congestion_pct": 45,
                "inference_ms": 1.45,
            })

        result = export_predictions_to_storage(records)
        return {
            "status": "success",
            "message": f"Exported {len(records)} prediction records to Firebase Storage",
            "storage": result,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction export to Firebase Storage failed: {e}",
        )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=False)

