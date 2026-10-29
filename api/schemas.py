"""
Pydantic schemas for GeoMind Traffic Forecasting API.
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


VALID_WEATHER = {
    "Clear", "Clouds", "Rain", "Snow", "Mist",
    "Fog", "Drizzle", "Thunderstorm", "Haze", "Smoke", "Squall"
}

VALID_HOLIDAYS = {
    "None", "Christmas Day", "Christmas Day (observed)",
    "New Years Day", "New Years Day (observed)",
    "Thanksgiving Day", "Labor Day", "Labor Day (observed)",
    "Independence Day", "Independence Day (observed)",
    "Memorial Day", "Martin Luther King Jr Day",
    "Columbus Day", "Veterans Day", "Veterans Day (observed)",
    "Washingtons Birthday",
}

VALID_ROAD_CONDITIONS = {
    "Dry / Normal", "Wet / Damp",
    "Snow / Ice Covered", "Slippery / Hazardous", "Flooded / Waterlogged"
}


class TrafficObservation(BaseModel):
    """One hour's contextual observation — no current traffic count needed."""

    date_time: str = Field(..., description="ISO 8601 timestamp", examples=["2026-10-01T08:00:00"])
    temp: float = Field(..., ge=200.0, le=330.0, description="Temperature in Kelvin")
    rain_1h: float = Field(default=0.0, ge=0.0, le=500.0, description="Rainfall mm/hr")
    snow_1h: float = Field(default=0.0, ge=0.0, le=100.0, description="Snowfall mm/hr")
    clouds_all: float = Field(default=0.0, ge=0.0, le=100.0, description="Cloud cover %")
    weather_main: str = Field(default="Clear", description="Primary weather category")
    road_condition: Optional[str] = Field(default="Dry / Normal")
    holiday: str = Field(default="None", description="US federal holiday or 'None'")

    @field_validator("date_time")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        if "T" in v and len(v) == 16:
            v = v + ":00"
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"date_time must be ISO 8601 like '2026-10-01T08:00:00', got: {v!r}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "date_time": "2026-10-01T08:00:00",
                "temp": 288.15, "rain_1h": 0.0, "snow_1h": 0.0,
                "clouds_all": 20.0, "weather_main": "Clear",
                "road_condition": "Dry / Normal", "holiday": "None",
            }
        }
    }


class PredictRequest(BaseModel):
    observation: TrafficObservation
    model_type: Literal["xgboost", "random_forest", "ridge_regression"] = Field(
        default="xgboost", description="ML model to use for prediction."
    )


class PredictResponse(BaseModel):
    status: str = "success"
    model_used: str
    input_datetime: str
    predicted_traffic_volume: float
    traffic_level: str
    congestion_pct: int
    summary: str
    unit: str = "vehicles/hr"
    inference_ms: float


class CompareItem(BaseModel):
    model: str
    predicted_traffic_volume: float
    traffic_level: str
    congestion_pct: int


class CompareResponse(BaseModel):
    status: str = "success"
    input_datetime: str
    weather: str
    results: List[CompareItem]


class HealthResponse(BaseModel):
    status: str
    service: str = "GeoMind"
    environment: str
    models_loaded: dict
    api_version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
