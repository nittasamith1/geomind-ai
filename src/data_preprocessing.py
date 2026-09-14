"""
GeoMind AI - Data Preprocessing Component
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Reusable, scikit-learn-based preprocessing pipeline:
1. Feature/Target separation.
2. Numerical standardization (StandardScaler).
3. Categorical encoding (OneHotEncoder with handle_unknown='ignore').
4. Strict Leakage Prevention: Fitted ONLY on training set; applied to validation/test/production inference.
5. Preprocessor artifact serialization via joblib.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException
from src.feature_engineering import engineer_features


class TrafficPreprocessor:
    """
    Encapsulates column transformations, scalers, and encoders.
    Enforces strict chronological fitting on training data only.
    """
    def __init__(self, artifact_path: str = "models/preprocessor.joblib"):
        self.artifact_path = Path(artifact_path)
        self.target_col = "future_traffic_volume"
        self.ignore_cols = ["date_time", "holiday", "weather_description", self.target_col]
        
        self.categorical_cols = ["weather_main"]
        self.pipeline: ColumnTransformer = None
        self.feature_names: List[str] = []

    def _determine_feature_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identifies numerical and categorical candidate features."""
        all_cols = [col for col in df.columns if col not in self.ignore_cols]
        cat_cols = [col for col in self.categorical_cols if col in all_cols]
        num_cols = [col for col in all_cols if col not in cat_cols]
        return num_cols, cat_cols

    def fit(self, train_df: pd.DataFrame) -> "TrafficPreprocessor":
        """
        Fits the ColumnTransformer strictly on the training partition.
        """
        try:
            logger.info("Fitting TrafficPreprocessor strictly on training data...")
            num_cols, cat_cols = self._determine_feature_columns(train_df)
            
            self.pipeline = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
                ],
                remainder="drop"
            )
            
            X_train_raw = train_df[num_cols + cat_cols]
            self.pipeline.fit(X_train_raw)
            
            # Extract output feature names
            cat_encoder = self.pipeline.named_transformers_["cat"]
            encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
            self.feature_names = num_cols + encoded_cat_names
            
            logger.info(f"TrafficPreprocessor fitted successfully across {len(self.feature_names)} features.")
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms input DataFrame into preprocessed feature matrix X and target vector y.
        """
        try:
            if self.pipeline is None:
                raise ValueError("TrafficPreprocessor has not been fitted! Call .fit() first or load saved artifact.")
            
            num_cols, cat_cols = self._determine_feature_columns(df)
            X_raw = df[num_cols + cat_cols]
            X_transformed = self.pipeline.transform(X_raw)
            
            y = df[self.target_col].to_numpy() if self.target_col in df.columns else None
            return X_transformed, y
        except Exception as e:
            raise CustomException(e, sys)

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience method to fit and transform on training data."""
        self.fit(train_df)
        return self.transform(train_df)

    def save_artifact(self) -> None:
        """Serializes fitted preprocessor to disk."""
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, self.artifact_path)
        logger.info(f"Preprocessor artifact saved to {self.artifact_path}")

    def load_artifact(self) -> "TrafficPreprocessor":
        """Loads serialized preprocessor from disk."""
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {self.artifact_path}")
        data = joblib.load(self.artifact_path)
        self.pipeline = data["pipeline"]
        self.feature_names = data["feature_names"]
        logger.info(f"Preprocessor artifact loaded from {self.artifact_path}")
        return self


def prepare_datasets() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    End-to-end dataset preparation pipeline:
    1. Loads train, val, test CSVs.
    2. Runs feature engineering on each partition independently.
    3. Fits preprocessor strictly on train; transforms val and test.
    4. Serializes preprocessor artifact.
    5. Returns (X_train, y_train, X_val, y_val, X_test, y_test, feature_names).
    """
    try:
        logger.info("Starting complete preprocessing pipeline...")
        train_raw = pd.read_csv("data/processed/train.csv")
        val_raw = pd.read_csv("data/processed/val.csv")
        test_raw = pd.read_csv("data/processed/test.csv")
        
        for d in [train_raw, val_raw, test_raw]:
            d["date_time"] = pd.to_datetime(d["date_time"])
            
        # Feature Engineering per split
        train_feat = engineer_features(train_raw)
        val_feat = engineer_features(val_raw)
        test_feat = engineer_features(test_raw)
        
        preprocessor = TrafficPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_feat)
        X_val, y_val = preprocessor.transform(val_feat)
        X_test, y_test = preprocessor.transform(test_feat)
        
        preprocessor.save_artifact()
        
        logger.info(
            f"Datasets prepared successfully: "
            f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor.feature_names
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_datasets()
    print("=" * 60)
    print("GeoMind AI: Preprocessing Pipeline Completed Successfully")
    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
    print(f"Total features: {len(feat_names)}")
    print(f"Feature Names: {feat_names[:8]} ... (+{len(feat_names)-8} more)")
    print("=" * 60)
