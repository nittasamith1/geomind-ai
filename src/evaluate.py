"""
GeoMind AI - Evaluation & Experiment Tracking Component
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Computes standardized evaluation metrics (MAE, RMSE, R2) and manages
the scientific experiment registry in 'experiments/results.csv'.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics for traffic volume forecasting.
    
    Returns:
        Dict with keys: 'mae', 'rmse', 'r2'
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }


def log_experiment(
    model_name: str,
    model_family: str,
    hyperparameters: Dict[str, Any],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    training_time_sec: float,
    results_path: str = "experiments/results.csv",
    sequence_length: int = 1,
    feature_set: str = "engineered_39_features",
    promoted: bool = False
) -> None:
    """
    Appends experiment execution parameters and benchmark results to the structured registry.
    """
    try:
        csv_path = Path(results_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        experiment_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name.replace(' ', '_').upper()}"
        timestamp = datetime.now().isoformat()
        
        row_data = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "model_family": model_family,
            "model_name": model_name,
            "sequence_length": sequence_length,
            "feature_set": feature_set,
            "hyperparameters": str(hyperparameters),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "training_time_sec": round(training_time_sec, 2),
            "promoted": promoted
        }
        
        df_new = pd.DataFrame([row_data])
        
        if csv_path.exists() and os.path.getsize(csv_path) > 0:
            df_new.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(csv_path, mode="w", header=True, index=False)
            
        logger.info(f"Experiment {experiment_id} logged to {results_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_results_leaderboard(results_path: str = "experiments/results.csv") -> pd.DataFrame:
    """Loads and formats the experiment leaderboard sorted by Validation MAE."""
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        return pd.DataFrame()
    df = pd.read_csv(results_path)
    return df.sort_values(by="val_mae").reset_index(drop=True)
