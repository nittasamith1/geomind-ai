"""
GeoMind AI - Data Ingestion & Integrity Component
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Responsibilities:
1. Ingest raw traffic telemetry from 'data/raw/Metro_Interstate_Traffic_Volume.csv'.
2. Perform comprehensive data integrity, schema, and sensor anomaly profiling.
3. Resolve timestamp multiplicity (duplicate hourly records from multiple weather reports).
4. Perform strict chronological train/val/test splitting (70% / 15% / 15%) without future leakage.
5. Export clean processed datasets to 'data/processed/'.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np

from src.logger import logger
from src.exception import CustomException


class DataIngestionConfig:
    """Configuration paths for data ingestion."""
    raw_data_path: Path = Path("data/raw/Metro_Interstate_Traffic_Volume.csv")
    processed_dir: Path = Path("data/processed")
    clean_data_path: Path = processed_dir / "traffic_clean.csv"
    train_data_path: Path = processed_dir / "train.csv"
    val_data_path: Path = processed_dir / "val.csv"
    test_data_path: Path = processed_dir / "test.csv"
    target_column: str = "traffic_volume"
    datetime_column: str = "date_time"
    temp_min_kelvin: float = 200.0  # Physical bound to catch 0.0 K sensor dropouts
    temp_max_kelvin: float = 330.0
    rain_max_mm: float = 100.0      # Physical bound to catch 9,831 mm telemetry spikes


class DataIngestion:
    """Executes data ingestion, profiling, and chronological splitting."""

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw CSV dataset and casts temporal index."""
        try:
            if not self.config.raw_data_path.exists():
                raise FileNotFoundError(f"Raw data file not found at: {self.config.raw_data_path}")
            
            logger.info(f"Loading raw dataset from {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)
            df[self.config.datetime_column] = pd.to_datetime(df[self.config.datetime_column])
            logger.info(f"Loaded raw dataset successfully with shape {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Conducts statistical, sensor anomaly, and temporal continuity profiling."""
        try:
            total_rows, total_cols = df.shape
            exact_duplicates = int(df.duplicated().sum())
            dt_col = self.config.datetime_column
            timestamp_duplicates = int(df[dt_col].duplicated().sum())
            unique_timestamps = int(df[dt_col].nunique())
            
            min_date = df[dt_col].min()
            max_date = df[dt_col].max()
            full_hourly_ticks = len(pd.date_range(start=min_date, end=max_date, freq="h"))
            missing_hours = full_hourly_ticks - unique_timestamps
            
            # Anomaly counts
            sub_200k_temp = int((df["temp"] < self.config.temp_min_kelvin).sum())
            extreme_rain = int((df["rain_1h"] > self.config.rain_max_mm).sum())
            
            profile = {
                "total_rows": total_rows,
                "total_cols": total_cols,
                "exact_duplicates": exact_duplicates,
                "timestamp_duplicates": timestamp_duplicates,
                "unique_timestamps": unique_timestamps,
                "start_time": str(min_date),
                "end_time": str(max_date),
                "expected_hours": full_hourly_ticks,
                "missing_hours": missing_hours,
                "temporal_completeness_pct": round((unique_timestamps / full_hourly_ticks) * 100, 2),
                "temp_sub_200k_count": sub_200k_temp,
                "rain_extreme_count": extreme_rain,
                "target_mean": float(df[self.config.target_column].mean()),
                "target_std": float(df[self.config.target_column].std())
            }
            return profile
        except Exception as e:
            raise CustomException(e, sys)

    def deduplicate_and_sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolves duplicate hourly records (caused by multiple weather logs per hour)
        and sanitizes hardware sensor dropouts.
        """
        try:
            dt_col = self.config.datetime_column
            target_col = self.config.target_column
            
            logger.info("Resolving hourly timestamp duplicates via scientific aggregation...")
            df_sorted = df.sort_values(by=dt_col)
            
            # Aggregate continuous telemetry by mean, categorical weather by first/mode
            agg_rules = {
                "holiday": "first",
                "temp": "mean",
                "rain_1h": "mean",
                "snow_1h": "mean",
                "clouds_all": "mean",
                "weather_main": "first",
                "weather_description": "first",
                target_col: "mean"
            }
            
            df_dedup = df_sorted.groupby(dt_col).agg(agg_rules).reset_index()
            df_dedup[target_col] = df_dedup[target_col].round().astype(int)
            
            # Sanitize unphysical sensor dropouts
            bad_temp_mask = (df_dedup["temp"] < self.config.temp_min_kelvin) | (df_dedup["temp"] > self.config.temp_max_kelvin)
            if bad_temp_mask.any():
                logger.info(f"Interpolating {bad_temp_mask.sum()} unphysical temperature anomalies (e.g. 0.0 Kelvin)...")
                df_dedup.loc[bad_temp_mask, "temp"] = np.nan
                df_dedup = df_dedup.set_index(dt_col)
                df_dedup["temp"] = df_dedup["temp"].interpolate(method="time").bfill().ffill()
                df_dedup = df_dedup.reset_index()
                
            bad_rain_mask = df_dedup["rain_1h"] > self.config.rain_max_mm
            if bad_rain_mask.any():
                logger.info(f"Capping {bad_rain_mask.sum()} extreme rain telemetry spikes (>100mm/h)...")
                df_dedup.loc[bad_rain_mask, "rain_1h"] = self.config.rain_max_mm

            logger.info(f"Sanitized & deduplicated dataset shape: {df_dedup.shape}")
            return df_dedup
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_chronologically(
        self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs strict chronological time-series splitting with mathematical non-overlap verification.
        """
        try:
            assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
            df_sorted = df.sort_values(by=self.config.datetime_column).reset_index(drop=True)
            n = len(df_sorted)
            
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_df = df_sorted.iloc[:train_end].copy()
            val_df = df_sorted.iloc[train_end:val_end].copy()
            test_df = df_sorted.iloc[val_end:].copy()
            
            # Strict mathematical leakage verification
            dt = self.config.datetime_column
            if train_df[dt].max() >= val_df[dt].min():
                raise ValueError(f"Temporal Leakage: Train max ({train_df[dt].max()}) >= Val min ({val_df[dt].min()})")
            if val_df[dt].max() >= test_df[dt].min():
                raise ValueError(f"Temporal Leakage: Val max ({val_df[dt].max()}) >= Test min ({test_df[dt].min()})")
                
            logger.info(f"Chronological Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (Zero Leakage Verified)")
            return train_df, val_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Tuple[Path, Path, Path]:
        """Runs end-to-end data ingestion pipeline."""
        try:
            self.config.processed_dir.mkdir(parents=True, exist_ok=True)
            
            df_raw = self.load_raw_data()
            profile = self.profile_data(df_raw)
            
            print("=" * 70)
            print(" GeoMind AI: Phase 1 — Dataset Ingestion & Scientific Profiling")
            print("=" * 70)
            print(f"Total Rows:                {profile['total_rows']:,}")
            print(f"Total Features:            {profile['total_cols']}")
            print(f"Exact Duplicates:          {profile['exact_duplicates']}")
            print(f"Duplicate Timestamps:      {profile['timestamp_duplicates']:,} (multiple weather logs/hr)")
            print(f"Unique Timestamps:         {profile['unique_timestamps']:,}")
            print(f"Temporal Span:             {profile['start_time']}  -->  {profile['end_time']}")
            print(f"Completeness Ratio:        {profile['temporal_completeness_pct']}% ({profile['missing_hours']:,} missing hours)")
            print(f"Sensor Anomalies:          Temp<200K: {profile['temp_sub_200k_count']}, Rain>100mm: {profile['rain_extreme_count']}")
            print("=" * 70)
            
            df_clean = self.deduplicate_and_sanitize(df_raw)
            df_clean.to_csv(self.config.clean_data_path, index=False)
            logger.info(f"Cleaned dataset saved to {self.config.clean_data_path}")
            
            train_df, val_df, test_df = self.split_data_chronologically(df_clean)
            train_df.to_csv(self.config.train_data_path, index=False)
            val_df.to_csv(self.config.val_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Saved Train to: {self.config.train_data_path}")
            logger.info(f"Saved Val to:   {self.config.val_data_path}")
            logger.info(f"Saved Test to:  {self.config.test_data_path}")
            
            return self.config.train_data_path, self.config.val_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
