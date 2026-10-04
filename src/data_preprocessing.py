"""
Data preprocessing: fit/transform preprocessor pipeline for GeoMind.

Features used are purely temporal + weather — no traffic_volume required
at inference time. This matches the frontend which collects only
weather/temporal inputs from the user.
"""

import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.logger import logger
from src.exception import CustomException
from src.feature_engineering import engineer_features


PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Columns to exclude from features (identifiers, raw input cols, target)
IGNORE_COLS = [
    "date_time", "holiday", "weather_description", "road_condition",
    "traffic_volume",          # raw input, not a feature
    "traffic_volume_target",   # the label
]

# Categorical columns to one-hot encode
CATEGORICAL_COLS = ["weather_main"]


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) from an engineered DataFrame."""
    all_cols = [c for c in df.columns if c not in IGNORE_COLS]
    cat_cols = [c for c in CATEGORICAL_COLS if c in all_cols]
    num_cols = [c for c in all_cols if c not in cat_cols]
    return num_cols, cat_cols


def fit_preprocessor(train_df: pd.DataFrame, save: bool = True, output_path: Path = None):
    """Fit ColumnTransformer on training data and optionally persist it."""
    num_cols, cat_cols = get_feature_columns(train_df)

    logger.info(f"Numeric features ({len(num_cols)}): {num_cols[:8]} ...")
    logger.info(f"Categorical features: {cat_cols}")

    pipeline = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    pipeline.fit(train_df[num_cols + cat_cols])

    cat_names = list(pipeline.named_transformers_["cat"].get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_names

    if save:
        target_path = output_path if output_path is not None else PREPROCESSOR_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "feature_names": feature_names}, target_path)
        logger.info(f"Preprocessor saved → {target_path} ({len(feature_names)} features)")

    return pipeline, feature_names


def transform(pipeline, df: pd.DataFrame):
    """Transform a DataFrame using the fitted preprocessor."""
    num_cols, cat_cols = get_feature_columns(df)
    X = pipeline.transform(df[num_cols + cat_cols])
    y = df["traffic_volume_target"].to_numpy() if "traffic_volume_target" in df.columns else None
    return X, y


def prepare_datasets() -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    List[str]
]:
    """Full pipeline: load CSVs → engineer features → fit preprocessor → split X/y."""
    try:
        logger.info("Loading processed CSVs ...")
        train_raw = pd.read_csv(PROCESSED_DIR / "train.csv")
        val_raw   = pd.read_csv(PROCESSED_DIR / "val.csv")
        test_raw  = pd.read_csv(PROCESSED_DIR / "test.csv")

        for d in [train_raw, val_raw, test_raw]:
            d["date_time"] = pd.to_datetime(d["date_time"])

        logger.info("Engineering features ...")
        train_feat = engineer_features(train_raw, is_training=True)
        val_feat   = engineer_features(val_raw,   is_training=True)
        test_feat  = engineer_features(test_raw,  is_training=True)

        logger.info("Fitting preprocessor ...")
        pipeline, feature_names = fit_preprocessor(train_feat, save=True)

        X_train, y_train = transform(pipeline, train_feat)
        X_val,   y_val   = transform(pipeline, val_feat)
        X_test,  y_test  = transform(pipeline, test_feat)

        logger.info(
            f"Dataset shapes — Train: {X_train.shape}, "
            f"Val: {X_val.shape}, Test: {X_test.shape}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_datasets()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features ({len(feat_names)}): {feat_names}")
