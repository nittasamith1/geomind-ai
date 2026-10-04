"""
Pydantic schemas for GeoMind Traffic Forecasting API.

The TrafficObservation schema accepts only what a user can realistically
provide: date/time, weather conditions, road condition, and holiday status.
Current traffic volume is intentionally excluded — the models predict
traffic based on temporal and weather patterns alone.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


VALID_WEATHER = {
    "Clear", "Clouds", "Rain", "Snow", "Mist",
    "Fog", "Drizzle", "Thunderstorm", "Haze", "Smoke", "Squall"
}

VALID_HOLIDAYS = {
    "None",
    "Christmas Day",
    "Christmas Day (observed)",
    "New Years Day",
    "New Years Day (observed)",
    "Thanksgiving Day",
    "Labor Day",
    "Labor Day (observed)",
    "Independence Day",
    "Independence Day (observed)",
    "Memorial Day",
    "Martin Luther King Jr Day",
    "Columbus Day",
    "Veterans Day",
    "Veterans Day (observed)",
    "Washingtons Birthday",
}

VALID_ROAD_CONDITIONS = {
    "Dry / Normal",
    "Wet / Damp",
    "Snow / Ice Covered",
    "Slippery / Hazardous",
    "Flooded / Waterlogged"
}


class TrafficObservation(BaseModel):
    """
    One hour's contextual observation used to predict traffic volume.
    All fields are user-providable or auto-detectable — no current traffic count needed.
    """

    date_time: str = Field(
        ...,
        description="Timestamp (ISO 8601: YYYY-MM-DDTHH:MM:SS)",
        examples=["2026-10-01T08:00:00"],
    )
    temp: float = Field(
        ...,
        ge=200.0,
        le=330.0,
        description="Temperature in Kelvin (200 K–330 K). 273.15 K = 0 °C, 293.15 K = 20 °C.",
    )
    rain_1h: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        description="Rainfall in the last hour (mm). 0 if none.",
    )
    snow_1h: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Snowfall in the last hour (mm). 0 if none.",
    )
    clouds_all: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Cloud cover percentage (0–100).",
    )
    weather_main: str = Field(
        default="Clear",
        description=(
            "Primary weather category. One of: "
            + ", ".join(sorted(VALID_WEATHER))
        ),
    )
    road_condition: Optional[str] = Field(
        default="Dry / Normal",
        description="Road surface condition (Dry / Normal, Wet / Damp, Snow / Ice Covered, etc.)",
    )
    holiday: str = Field(
        default="None",
        description="US federal holiday name, or 'None' if not a holiday.",
    )

    @field_validator("date_time")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        try:
            # Handle standard ISO formats, including with or without seconds
            if "T" in v and len(v) == 16:
                v = v + ":00"
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(
                f"date_time must be ISO 8601 format like '2026-10-01T08:00:00', got: {v!r}"
            )
        return v

    @field_validator("weather_main")
    @classmethod
    def validate_weather(cls, v: str) -> str:
        if v not in VALID_WEATHER:
            return v
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "date_time": "2026-10-01T08:00:00",
                "temp": 288.15,
                "rain_1h": 0.0,
                "snow_1h": 0.0,
                "clouds_all": 20.0,
                "weather_main": "Clear",
                "road_condition": "Dry / Normal",
                "holiday": "None",
            }
        }
    }


class PredictRequest(BaseModel):
    observation: TrafficObservation
    model_type: Literal["xgboost", "random_forest", "ridge_regression"] = Field(
        default="xgboost",
        description="Which ML model to use for prediction.",
    )


class PredictResponse(BaseModel):
    status: str = "success"
    model_used: str
    input_datetime: str
    predicted_traffic_volume: float
    traffic_level: str          # Very Low / Low / Moderate / High / Very High / Severe
    congestion_pct: int         # 0% - 100% capacity estimate
    summary: str                # Contextual description of conditions
    unit: str = "vehicles/hr"
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str = "GeoMind"
    environment: str = "production"
    models_loaded: dict
    api_version: str = "3.1.0"


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
