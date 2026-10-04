"""
Feature engineering for GeoMind Traffic Forecasting.

DESIGN PRINCIPLE: All features are derived exclusively from:
  - Date & time (temporal features, cyclical encodings, rush-hour flags)
  - Weather inputs (temp, rain_1h, snow_1h, clouds_all, weather_main)
  - Holiday flag

We deliberately exclude lag/rolling features that require historical
traffic_volume because the end-user does NOT know the current traffic
count. The model must predict traffic purely from contextual signals.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException


def engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Build all features from date_time + weather + holiday inputs only.

    Parameters
    ----------
    df          : DataFrame with columns [date_time, temp, rain_1h, snow_1h,
                  clouds_all, weather_main, holiday] and optionally
                  [traffic_volume] (only needed during training to create target).
    is_training : If True, creates target column 'traffic_volume_target'
                  from the raw traffic_volume column and drops rows without it.

    Returns
    -------
    Engineered DataFrame ready for preprocessor.
    """
    try:
        df = df.sort_values("date_time").reset_index(drop=True).copy()
        df["date_time"] = pd.to_datetime(df["date_time"])

        # ── Target (training only) ──────────────────────────────────────
        if is_training and "traffic_volume" in df.columns:
            df["traffic_volume_target"] = df["traffic_volume"]

        # ── Temporal features ───────────────────────────────────────────
        dt = df["date_time"]
        df["hour"]        = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek       # Mon=0, Sun=6
        df["month"]       = dt.dt.month
        df["year"]        = dt.dt.year
        df["day_of_year"] = dt.dt.dayofyear
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["quarter"]     = dt.dt.quarter

        # Binary flags
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

        # CRITICAL FIX: the raw dataset uses NaN for non-holiday rows.
        # Normalize holiday column before comparison so NaN → "None".
        # This must be done before is_holiday is computed to avoid every
        # row being falsely flagged as a holiday (NaN != "None" is True).
        df["holiday"] = df["holiday"].fillna("None").astype(str).str.strip()
        df["is_holiday"] = (~df["holiday"].isin(["None", "none", "", "nan"])).astype(int)

        # Cyclical encoding (preserves circular nature of time)
        df["sin_hour"]    = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["cos_hour"]    = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["sin_dow"]     = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["cos_dow"]     = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["sin_month"]   = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"]   = np.cos(2 * np.pi * df["month"] / 12.0)
        df["sin_doy"]     = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
        df["cos_doy"]     = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

        # Rush hour buckets (weekday only)
        #   0 = off-peak | 1 = morning rush (7-9am) | 2 = evening rush (4-7pm)
        #   3 = midday    | 4 = overnight
        def _rush_bucket(row):
            if row["is_weekend"] or row["is_holiday"]:
                return 0
            h = row["hour"]
            if 7 <= h <= 9:
                return 1
            if 16 <= h <= 18:
                return 2
            if 10 <= h <= 15:
                return 3
            if 0 <= h <= 5:
                return 4
            return 0

        df["rush_bucket"]  = df.apply(_rush_bucket, axis=1)
        df["is_rush_hour"] = (df["rush_bucket"].isin([1, 2])).astype(int)

        # Part-of-day (0=night, 1=morning, 2=midday, 3=afternoon, 4=evening)
        df["part_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 5, 10, 13, 17, 23],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        # ── Weather-derived features ────────────────────────────────────
        # Convert Kelvin → Celsius for interpretability
        df["temp_celsius"] = df["temp"] - 273.15

        # Comfort zone: 15-25°C encourages driving
        df["temp_comfortable"] = ((df["temp_celsius"] >= 15) &
                                  (df["temp_celsius"] <= 25)).astype(int)

        # Extreme cold / hot flags
        df["temp_extreme_cold"] = (df["temp_celsius"] < 0).astype(int)
        df["temp_extreme_hot"]  = (df["temp_celsius"] > 35).astype(int)

        # Precipitation severity (combined)
        df["has_rain"]   = (df["rain_1h"] > 0).astype(int)
        df["has_snow"]   = (df["snow_1h"] > 0).astype(int)
        df["rain_heavy"] = (df["rain_1h"] > 5).astype(int)
        df["snow_heavy"] = (df["snow_1h"] > 0.5).astype(int)
        df["precip_total"] = df["rain_1h"] + df["snow_1h"]

        # Cloud cover buckets
        df["mostly_clear"] = (df["clouds_all"] <= 25).astype(int)
        df["overcast"]     = (df["clouds_all"] >= 75).astype(int)

        # Weather severity (0=mild, 1=moderate, 2=severe)
        severe_weather = {"Snow", "Thunderstorm"}
        moderate_weather = {"Rain", "Drizzle", "Mist", "Fog", "Haze", "Smoke"}

        def _weather_severity(w):
            if w in severe_weather:
                return 2
            if w in moderate_weather:
                return 1
            return 0

        df["weather_severity"] = df["weather_main"].apply(_weather_severity)

        # Bad-weather compound flag (likely to reduce traffic)
        df["bad_weather"] = (
            (df["weather_severity"] >= 1) |
            (df["rain_heavy"] == 1) |
            (df["snow_heavy"] == 1)
        ).astype(int)

        # Incorporate road condition surface effects if specified
        if "road_condition" in df.columns:
            for idx, r_cond in df["road_condition"].items():
                rc_str = str(r_cond).lower()
                if "snow" in rc_str or "ice" in rc_str or "hazardous" in rc_str:
                    df.at[idx, "has_snow"] = 1
                    df.at[idx, "weather_severity"] = max(df.at[idx, "weather_severity"], 2)
                    df.at[idx, "bad_weather"] = 1
                elif "wet" in rc_str or "damp" in rc_str or "slippery" in rc_str:
                    df.at[idx, "has_rain"] = 1
                    df.at[idx, "bad_weather"] = 1
                elif "flood" in rc_str or "water" in rc_str:
                    df.at[idx, "rain_heavy"] = 1
                    df.at[idx, "weather_severity"] = 2
                    df.at[idx, "bad_weather"] = 1


        # ── Interaction features ────────────────────────────────────────
        # Rush hour + bad weather interaction
        df["rush_bad_weather"]  = df["is_rush_hour"] * df["bad_weather"]
        # Rush hour on weekday
        df["rush_weekday"]      = df["is_rush_hour"] * (1 - df["is_weekend"])
        # Holiday morning commute suppression
        df["holiday_rush"]      = df["is_holiday"] * df["is_rush_hour"]
        # Weekend midday activity
        df["weekend_midday"]    = df["is_weekend"] * (df["part_of_day"] == 2).astype(int)
        # Summer weekend
        df["summer_weekend"]    = df["is_weekend"] * (df["month"].isin([6, 7, 8])).astype(int)

        # ── Drop rows without target in training mode ───────────────────
        if is_training and "traffic_volume_target" in df.columns:
            df = df.dropna(subset=["traffic_volume_target"]).reset_index(drop=True)

        # Fill any remaining NaNs in numeric columns with column median
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        logger.info(f"Feature engineering done: {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    train["date_time"] = pd.to_datetime(train["date_time"])
    result = engineer_features(train, is_training=True)
    print(result[["date_time", "hour", "traffic_volume_target"]].head(10))
    print(f"Features: {result.shape[1]}, Rows: {result.shape[0]}")
