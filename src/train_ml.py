"""
GeoMind AI - Traditional Machine Learning & Baseline Training Pipeline
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Trains and rigorously compares:
1. Mean Baseline
2. Persistence Baseline (traffic_volume[t])
3. Linear Regression (Ridge)
4. Random Forest Regressor
5. XGBoost Regressor

Tracks experiment metadata, metrics (MAE, RMSE, R2), training runtime,
and serializes winning models to 'models/ml/'.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException
from src.feature_engineering import engineer_features
from src.data_preprocessing import prepare_datasets
from src.evaluate import compute_metrics, log_experiment


class MeanBaseline:
    """Predicts historical mean of training target."""
    def __init__(self):
        self.mean_val = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_val = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(X.shape[0],), fill_value=self.mean_val)


class PersistenceBaseline:
    """Predicts next-hour traffic volume equals current hour traffic volume: y_pred(t+1) = traffic_volume(t)."""
    def __init__(self, current_traffic_feature_idx: int):
        self.feat_idx = current_traffic_feature_idx
        self.scaler_mean = 0.0
        self.scaler_scale = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, scaler=None):
        if scaler is not None:
            self.scaler_mean = scaler.mean_[self.feat_idx]
            self.scaler_scale = scaler.scale_[self.feat_idx]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Invert standardization to recover raw vehicle count: X_raw = X * scale + mean
        return X[:, self.feat_idx] * self.scaler_scale + self.scaler_mean


def run_ml_training_pipeline() -> pd.DataFrame:
    """Executes the full baseline and traditional ML training, evaluation, and logging pipeline."""
    try:
        logger.info("Initializing Traditional ML Training Pipeline...")
        X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_datasets()
        
        # Load fitted preprocessor scaler for Persistence baseline inversion
        preprocessor_data = joblib.load("models/preprocessor.joblib")
        pipeline = preprocessor_data["pipeline"]
        scaler = pipeline.named_transformers_["num"]
        traffic_vol_idx = feat_names.index("traffic_volume")
        
        models_to_train = {
            "Mean Baseline": {
                "family": "Baseline",
                "model": MeanBaseline(),
                "params": {"strategy": "global_mean"}
            },
            "Persistence Baseline": {
                "family": "Baseline",
                "model": PersistenceBaseline(traffic_vol_idx),
                "params": {"strategy": "t+1_equals_t"}
            },
            "Linear Regression": {
                "family": "Linear",
                "model": Ridge(alpha=1.0, random_state=42),
                "params": {"alpha": 1.0, "solver": "auto"}
            },
            "Random Forest": {
                "family": "Tree Ensemble",
                "model": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
                "params": {"n_estimators": 100, "max_depth": 16}
            },
            "XGBoost": {
                "family": "Gradient Boosted Trees",
                "model": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1),
                "params": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08}
            }
        }
        
        leaderboard_rows = []
        save_dir = Path("models/ml")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, spec in models_to_train.items():
            logger.info(f"--- Training & Evaluating: {name} ---")
            clf = spec["model"]
            params = spec["params"]
            family = spec["family"]
            
            start_time = time.time()
            if name == "Persistence Baseline":
                clf.fit(X_train, y_train, scaler=scaler)
            else:
                clf.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict across all splits
            pred_train = clf.predict(X_train)
            pred_val = clf.predict(X_val)
            pred_test = clf.predict(X_test)
            
            train_m = compute_metrics(y_train, pred_train)
            val_m = compute_metrics(y_val, pred_val)
            test_m = compute_metrics(y_test, pred_test)
            
            logger.info(f"{name} -> Val MAE: {val_m['mae']} | Val RMSE: {val_m['rmse']} | Val R2: {val_m['r2']} (Time: {train_time:.2f}s)")
            
            # Save artifact if not a baseline
            if family != "Baseline":
                model_filename = f"{name.lower().replace(' ', '_')}.joblib"
                artifact_path = save_dir / model_filename
                joblib.dump(clf, artifact_path)
                logger.info(f"Saved model artifact to {artifact_path}")
                
            # Log to registry
            log_experiment(
                model_name=name,
                model_family=family,
                hyperparameters=params,
                train_metrics=train_m,
                val_metrics=val_m,
                test_metrics=test_m,
                training_time_sec=train_time,
                sequence_length=1,
                promoted=(name == "XGBoost")
            )
            
            leaderboard_rows.append({
                "Model": name,
                "Family": family,
                "Val MAE": val_m["mae"],
                "Val RMSE": val_m["rmse"],
                "Val R2": val_m["r2"],
                "Test MAE": test_m["mae"],
                "Test RMSE": test_m["rmse"],
                "Test R2": test_m["r2"],
                "Train Time (s)": round(train_time, 2)
            })
            
        leaderboard = pd.DataFrame(leaderboard_rows).sort_values(by="Val MAE").reset_index(drop=True)
        return leaderboard
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df_leaderboard = run_ml_training_pipeline()
    print("\n" + "=" * 80)
    print(" GeoMind AI: Traditional ML & Baseline Benchmark Leaderboard")
    print("=" * 80)
    print(df_leaderboard.to_string(index=False))
    print("=" * 80)
