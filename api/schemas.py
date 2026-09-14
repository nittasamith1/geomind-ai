"""
GeoMind AI - FastAPI Request & Response Schemas
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Pydantic v2 models for strict input validation and structured API responses.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Request Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TrafficObservation(BaseModel):
    """
    Single hourly observation for ML model inference.
    Matches the schema from Metro Interstate Traffic Volume dataset.
    """
    date_time: str = Field(
        ...,
        description="Timestamp in ISO format (YYYY-MM-DDTHH:MM:SS)",
        examples=["2024-08-15T08:00:00"]
    )
    temp: float = Field(
        ..., ge=200.0, le=330.0,
        description="Atmospheric temperature in Kelvin (valid: 200-330K)"
    )
    rain_1h: float = Field(
        default=0.0, ge=0.0,
        description="Rainfall in the past hour (mm)"
    )
    snow_1h: float = Field(
        default=0.0, ge=0.0,
        description="Snowfall in the past hour (mm)"
    )
    clouds_all: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cloud coverage percentage (0-100)"
    )
    weather_main: str = Field(
        ...,
        description="Primary weather category (e.g., Clear, Rain, Snow, Clouds, Mist)"
    )
    holiday: str = Field(
        default="None",
        description="US holiday name if applicable, else 'None'"
    )
    traffic_volume: float = Field(
        ..., ge=0.0, le=10000.0,
        description="Current hour observed traffic volume (vehicles/hr)"
    )

    @field_validator("date_time")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"date_time must be ISO format (YYYY-MM-DDTHH:MM:SS), got: {v!r}")
        return v

    @field_validator("weather_main")
    @classmethod
    def validate_weather_main(cls, v: str) -> str:
        allowed = {"Clear", "Clouds", "Rain", "Snow", "Mist", "Fog", "Drizzle",
                   "Thunderstorm", "Haze", "Smoke", "Squall", "Dust", "Sand", "Ash"}
        if v not in allowed:
            # Normalize common variants
            v = v.strip().title()
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "date_time": "2024-08-15T08:00:00",
            "temp": 287.5,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40.0,
            "weather_main": "Clear",
            "holiday": "None",
            "traffic_volume": 4200.0
        }
    }}


class SinglePredictionRequest(BaseModel):
    """Request schema for one-step ahead (t+1) traffic forecasting."""
    observation: TrafficObservation
    model_type: Literal["xgboost", "random_forest", "lstm", "gru"] = Field(
        default="xgboost",
        description="Which trained model to use for inference"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch inference over multiple sequential observations."""
    observations: List[TrafficObservation] = Field(
        ..., min_length=1, max_length=500,
        description="List of 1-500 hourly observations"
    )
    model_type: Literal["xgboost", "random_forest", "lstm", "gru"] = Field(
        default="xgboost",
        description="Which trained model to use for inference"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Single prediction result with confidence metadata."""
    input_datetime: str = Field(description="Input timestamp")
    predicted_traffic_volume: float = Field(description="Predicted next-hour traffic volume (vehicles/hr)")
    model_used: str = Field(description="Model identifier used for this prediction")
    forecast_horizon: str = Field(default="t+1 hour", description="Prediction horizon")


class SinglePredictionResponse(BaseModel):
    """API response for single prediction endpoint."""
    status: str = Field(default="success")
    prediction: PredictionResult
    inference_time_ms: float = Field(description="Model inference latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """API response for batch prediction endpoint."""
    status: str = Field(default="success")
    predictions: List[PredictionResult]
    total_predictions: int
    inference_time_ms: float


class ModelInfoResponse(BaseModel):
    """Metadata about available trained models."""
    available_models: List[str]
    default_model: str
    best_model: str
    best_model_test_mae: float
    best_model_test_r2: float
    feature_count: int
    training_dataset: str
    target_variable: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    models_loaded: Dict[str, bool]
    api_version: str = "1.0.0"
    project: str = "GeoMind AI — Urban Traffic Forecasting"


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    status: str = "error"
    error_type: str
    message: str
    detail: Optional[Any] = None
