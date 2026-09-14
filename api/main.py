"""
GeoMind AI - FastAPI Application Entry Point
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Production-ready REST API serving trained ML/DL traffic forecasting models.

Endpoints:
    GET  /                    → Landing page + API summary
    GET  /health              → Health check (model load status)
    GET  /models              → Available models & performance metadata
    POST /predict             → Single observation → next-hour traffic forecast
    POST /predict/batch       → Batch inference (up to 500 observations)
    GET  /docs                → Interactive Swagger UI (auto-generated)
    GET  /redoc               → ReDoc documentation

Design Principles:
    - Zero-downtime model loading at startup (lifespan context manager)
    - Strict Pydantic v2 input validation
    - Structured JSON error responses
    - Separation of inference logic from API routing (api/prediction.py)
    - Full request/response schema documentation
"""

import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from api.schemas import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResult,
    ModelInfoResponse,
    HealthResponse,
    ErrorResponse,
)
from api.prediction import (
    load_all_models,
    predict_ml,
    predict_dl,
    get_loaded_model_status,
    ML_MODEL_PATHS,
    DL_MODEL_CONFIGS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan: Model pre-loading at startup
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all model artifacts at startup and logs their status.
    Prevents cold-start latency on first prediction request.
    """
    logger.info("=" * 60)
    logger.info("GeoMind AI API — Startup: Loading model artifacts...")
    logger.info("=" * 60)
    status_map = load_all_models()
    for model_name, ok in status_map.items():
        status_str = "LOADED" if ok else "FAILED"
        marker = "[OK]" if ok else "[XX]"
        logger.info(f"  {marker} {model_name}: {status_str}")
    logger.info("GeoMind AI API ready to serve predictions.")
    yield
    logger.info("GeoMind AI API — Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GeoMind AI — Urban Traffic Forecasting API",
    description="""
## GeoMind AI: Self-Learning Urban Traffic Forecasting System

**Production ML/DL REST API** for next-hour traffic volume prediction.

### Models Available
| Model | Type | Test MAE | Test R² |
|-------|------|----------|---------|
| XGBoost | Gradient Boosted Trees | ~130 veh/hr | ~0.981 |
| Random Forest | Tree Ensemble | ~180 veh/hr | ~0.970 |
| LSTM (L=6) | Deep Learning | ~212 veh/hr | ~0.969 |
| GRU (L=12) | Deep Learning | ~224 veh/hr | ~0.971 |

### Key Features
- **39 engineered features**: temporal cyclicals, autoregressive lags, rolling statistics
- **Strict data leakage prevention**: preprocessor fitted only on training data
- **Chronological train/val/test split**: 2012-2016 / 2017 / 2018
- **Pydantic v2 validation**: all inputs strictly validated before inference
    """,
    version="1.0.0",
    contact={
        "name": "Applied Scientist Candidate",
        "url": "https://github.com/geomind-ai",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# CORS — allow any origin for development / demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Global Exception Handler
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_type=type(exc).__name__,
            message=str(exc)
        ).model_dump(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Landing page — API summary and available endpoints."""
    return {
        "project": "GeoMind AI — Urban Traffic Forecasting System",
        "description": "Production ML/DL API for next-hour traffic volume prediction",
        "version": "1.0.0",
        "target_role": "Amazon Applied Scientist I Intern",
        "endpoints": {
            "health":        "GET  /health",
            "models_info":   "GET  /models",
            "single_predict":"POST /predict",
            "batch_predict": "POST /predict/batch",
            "swagger_ui":    "GET  /docs",
            "redoc":         "GET  /redoc",
        },
        "dataset": "Metro Interstate Traffic Volume (UCI / MnDOT ATR Station 301)",
        "task": "One-step-ahead regression: predict traffic_volume(t+1) from features at t",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check — model load status"
)
async def health_check():
    """
    Returns API health status and per-model load status.
    Use this to verify all artifacts loaded successfully at startup.
    """
    models_status = get_loaded_model_status()
    all_healthy = all(models_status.values())
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        models_loaded=models_status,
    )


@app.get(
    "/models",
    response_model=ModelInfoResponse,
    tags=["Model Info"],
    summary="Available models and performance metadata"
)
async def model_info():
    """
    Returns metadata about all trained models including best-performing model,
    evaluation metrics, and feature set information.
    """
    return ModelInfoResponse(
        available_models=["xgboost", "random_forest", "lstm", "gru"],
        default_model="xgboost",
        best_model="xgboost",
        best_model_test_mae=130.0,
        best_model_test_r2=0.981,
        feature_count=39,
        training_dataset="Metro Interstate Traffic Volume (UCI ML Repo)",
        target_variable="future_traffic_volume (next-hour vehicles/hr)",
    )


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Prediction"],
    summary="Predict next-hour traffic volume (single observation)",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    }
)
async def predict_single(request: SinglePredictionRequest):
    """
    **Next-Hour Traffic Forecasting — Single Observation**

    Accepts one hourly traffic observation and returns the predicted
    traffic volume for the following hour.

    **Supported models:** `xgboost` (default), `random_forest`, `lstm`, `gru`

    > **Note:** LSTM/GRU models accept a single observation but use lag features
    > internally from the preprocessed feature vector. For true sequential inference
    > with context windows, use `POST /predict/batch`.
    """
    t0 = time.perf_counter()
    model_type = request.model_type
    obs_dict = request.observation.model_dump()

    try:
        if model_type in ML_MODEL_PATHS:
            predicted_volume = predict_ml(model_type, obs_dict)
        elif model_type in DL_MODEL_CONFIGS:
            # Single-observation DL: wrap in a list (will use lag features from engineered features)
            predicted_volume = predict_dl(model_type, [obs_dict])
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model_type '{model_type}'. Choose from: xgboost, random_forest, lstm, gru"
            )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    inference_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Prediction [{model_type}]: {predicted_volume:.0f} veh/hr | {inference_ms:.2f}ms")

    return SinglePredictionResponse(
        prediction=PredictionResult(
            input_datetime=request.observation.date_time,
            predicted_traffic_volume=predicted_volume,
            model_used=model_type,
        ),
        inference_time_ms=round(inference_ms, 3),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Batch traffic volume prediction (multiple sequential observations)",
    responses={
        200: {"description": "Successful batch predictions"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"description": "Validation error"},
    }
)
async def predict_batch(request: BatchPredictionRequest):
    """
    **Batch Traffic Forecasting**

    Accepts a list of sequential hourly observations (oldest → newest)
    and returns a prediction for each timestep.

    - **ML models** (xgboost, random_forest): Each observation predicted independently.
    - **DL models** (lstm, gru): Uses a rolling context window over the sequence.
      Requires at least `seq_len` observations (LSTM: 6, GRU: 12).

    Maximum batch size: **500 observations**.
    """
    t0 = time.perf_counter()
    model_type = request.model_type
    observations = [obs.model_dump() for obs in request.observations]

    predictions_out: List[PredictionResult] = []

    try:
        if model_type in ML_MODEL_PATHS:
            # ML: each observation independently
            for obs_dict in observations:
                vol = predict_ml(model_type, obs_dict)
                predictions_out.append(PredictionResult(
                    input_datetime=obs_dict["date_time"],
                    predicted_traffic_volume=vol,
                    model_used=model_type,
                ))
        elif model_type in DL_MODEL_CONFIGS:
            seq_len = DL_MODEL_CONFIGS[model_type]["seq_len"] if model_type in DL_MODEL_CONFIGS else 6
            # DL: rolling window inference
            for i in range(len(observations)):
                context = observations[max(0, i - seq_len + 1): i + 1]
                vol = predict_dl(model_type, context)
                predictions_out.append(PredictionResult(
                    input_datetime=observations[i]["date_time"],
                    predicted_traffic_volume=vol,
                    model_used=model_type,
                ))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model_type '{model_type}'"
            )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    inference_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Batch prediction [{model_type}]: {len(predictions_out)} preds | {inference_ms:.2f}ms")

    return BatchPredictionResponse(
        predictions=predictions_out,
        total_predictions=len(predictions_out),
        inference_time_ms=round(inference_ms, 3),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
    )
