"""
GeoMind AI - Feature Engineering Component
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Transforms raw traffic telemetry and timestamps into rich predictive feature vectors:
1. Target generation: future_traffic_volume = traffic_volume(t + 1 hr) with temporal continuity verification.
2. Temporal & Cyclical encoding: sin/cos transformations for hour and day_of_week.
3. Autoregressive Lags: lag_1, lag_2, lag_3, lag_6, lag_12, lag_24.
4. Rolling Window Statistics: moving averages and volatilities (3h, 6h, 24h).
5. Regime Indicators: rush hour, weekend, and holiday indicators.
"""

import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException


def create_target_variable(df: pd.DataFrame, datetime_col: str = "date_time", target_col: str = "traffic_volume") -> pd.DataFrame:
    """
    Constructs next-hour target variable: future_traffic_volume = traffic_volume(t + 1).
    Enforces strict temporal continuity: if next row is NOT exactly (t + 1 hour), target is set to NaN.
    """
    df_out = df.sort_values(by=datetime_col).copy()
    
    # Check temporal delta to ensure t+1 is genuinely 1 hour ahead
    time_diff_next = df_out[datetime_col].shift(-1) - df_out[datetime_col]
    one_hour = pd.Timedelta(hours=1)
    
    raw_next_traffic = df_out[target_col].shift(-1)
    # Mask out jumps where data logging was interrupted (> 1 hour gap)
    df_out["future_traffic_volume"] = np.where(time_diff_next == one_hour, raw_next_traffic, np.nan)
    
    valid_targets = df_out["future_traffic_volume"].notnull().sum()
    logger.info(f"Target 'future_traffic_volume' constructed. Valid continuous (t+1) targets: {valid_targets:,} / {len(df_out):,}")
    return df_out


def add_temporal_features(df: pd.DataFrame, datetime_col: str = "date_time") -> pd.DataFrame:
    """
    Extracts calendar and cyclical temporal representations.
    """
    df_out = df.copy()
    dt = df_out[datetime_col]
    
    df_out["hour"] = dt.dt.hour
    df_out["day_of_week"] = dt.dt.dayofweek
    df_out["month"] = dt.dt.month
    df_out["year"] = dt.dt.year
    df_out["is_weekend"] = (df_out["day_of_week"] >= 5).astype(int)
    
    # Cyclical trigonometric encodings: preserving circular distance (e.g. 23:00 to 00:00)
    df_out["sin_hour"] = np.sin(2 * np.pi * df_out["hour"] / 24.0)
    df_out["cos_hour"] = np.cos(2 * np.pi * df_out["hour"] / 24.0)
    
    df_out["sin_day_of_week"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
    df_out["cos_day_of_week"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)
    
    # Commuter rush hour indicator: Weekdays 07:00-09:00 & 16:00-18:00
    morning_rush = (df_out["hour"] >= 7) & (df_out["hour"] <= 9)
    evening_rush = (df_out["hour"] >= 16) & (df_out["hour"] <= 18)
    df_out["is_rush_hour"] = ((morning_rush | evening_rush) & (df_out["is_weekend"] == 0)).astype(int)
    
    # Public holiday flag
    df_out["is_holiday"] = (df_out["holiday"] != "None").astype(int)
    
    return df_out


def add_autoregressive_lags(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 6, 12, 24],
    target_col: str = "traffic_volume",
    datetime_col: str = "date_time"
) -> pd.DataFrame:
    """
    Constructs autoregressive lag features: lag_k = traffic_volume(t - k).
    Verifies temporal continuity: if time delta != k hours, lag is NaN.
    """
    df_out = df.copy()
    
    for k in lags:
        lag_time_diff = df_out[datetime_col] - df_out[datetime_col].shift(k)
        expected_diff = pd.Timedelta(hours=k)
        raw_lag = df_out[target_col].shift(k)
        df_out[f"traffic_lag_{k}"] = np.where(lag_time_diff == expected_diff, raw_lag, np.nan)
        
    return df_out


def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [3, 6, 24],
    target_col: str = "traffic_volume"
) -> pd.DataFrame:
    """
    Constructs rolling statistics (mean and standard deviation) over past observed traffic.
    Uses closed='left' so the current target step is strictly excluded from rolling calculation (preventing leakage).
    """
    df_out = df.copy()
    
    for w in windows:
        # Shift by 1 first to strictly avoid leaking current observation t into rolling window
        rolled = df_out[target_col].shift(1).rolling(window=w, min_periods=max(2, w // 2))
        df_out[f"rolling_mean_{w}h"] = rolled.mean()
        df_out[f"rolling_std_{w}h"] = rolled.std()
        
    return df_out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the comprehensive feature engineering pipeline on a DataFrame.
    """
    try:
        logger.info("Executing feature engineering pipeline...")
        df_feat = create_target_variable(df)
        df_feat = add_temporal_features(df_feat)
        df_feat = add_autoregressive_lags(df_feat)
        df_feat = add_rolling_features(df_feat)
        
        # Drop rows where target or key lag_1 is NaN due to boundaries/gaps
        initial_len = len(df_feat)
        df_feat = df_feat.dropna(subset=["future_traffic_volume", "traffic_lag_1"]).reset_index(drop=True)
        
        # Fill remaining long-lag NaNs (e.g. lag_24 in beginning of segments) with forward-fill or median
        df_feat = df_feat.bfill().ffill()
        
        dropped = initial_len - len(df_feat)
        logger.info(f"Feature engineering completed: {len(df_feat):,} usable rows ({dropped:,} boundary rows dropped). Total features: {df_feat.shape[1]}")
        return df_feat
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    train_raw = pd.read_csv("data/processed/train.csv")
    train_raw["date_time"] = pd.to_datetime(train_raw["date_time"])
    feat_df = engineer_features(train_raw)
    print("Engineered Features Sample:")
    print(feat_df[["date_time", "traffic_volume", "traffic_lag_1", "future_traffic_volume", "sin_hour", "cos_hour", "is_rush_hour"]].head())
