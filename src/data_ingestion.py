import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.logger import logger
from src.exception import CustomException


RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH   = PROCESSED_DIR / "val.csv"
TEST_PATH  = PROCESSED_DIR / "test.csv"


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info(f"Loaded raw data: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date_time")

    # ── CRITICAL: normalise holiday field ──────────────────────────────
    # The raw CSV uses NaN for non-holiday rows. Filling to the literal
    # string "None" lets feature_engineering correctly detect holidays.
    # Without this fix ALL rows are mistakenly treated as holidays because
    # NaN != 'None' evaluates to True in Python.
    df["holiday"] = df["holiday"].fillna("None")

    # Aggregate duplicated timestamps (multiple weather reports per hour)
    agg_rules = {
        "holiday": "first",
        "temp": "mean",
        "rain_1h": "mean",
        "snow_1h": "mean",
        "clouds_all": "mean",
        "weather_main": "first",
        "weather_description": "first",
        "traffic_volume": "mean",
    }
    df = df.groupby("date_time").agg(agg_rules).reset_index()
    df["traffic_volume"] = df["traffic_volume"].round().astype(int)

    # Fix bad temperature readings (0 Kelvin sensor dropout)
    bad_temp = (df["temp"] < 200) | (df["temp"] > 330)
    if bad_temp.any():
        df.loc[bad_temp, "temp"] = np.nan
        df = df.set_index("date_time")
        df["temp"] = df["temp"].interpolate(method="time").bfill().ffill()
        df = df.reset_index()

    # Cap extreme rainfall spikes
    df.loc[df["rain_1h"] > 100, "rain_1h"] = 100.0

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    df = df.sort_values("date_time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    logger.info(f"Split -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def run_data_ingestion():
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df = load_raw_data()
        df = clean_data(df)
        train, val, test = split_data(df)

        train.to_csv(TRAIN_PATH, index=False)
        val.to_csv(VAL_PATH, index=False)
        test.to_csv(TEST_PATH, index=False)

        print(f"Data ingestion complete.")
        print(f"  Train: {len(train)} rows -> {TRAIN_PATH}")
        print(f"  Val:   {len(val)} rows   -> {VAL_PATH}")
        print(f"  Test:  {len(test)} rows  -> {TEST_PATH}")
        return TRAIN_PATH, VAL_PATH, TEST_PATH
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_data_ingestion()
