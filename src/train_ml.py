"""
ML training pipeline for GeoMind Traffic Forecasting.

Models predict traffic_volume from purely temporal + weather features.
No current traffic count is required at prediction time.

Hyperparameters are tuned for the Metro Interstate dataset (~48k rows).
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.logger import logger
from src.exception import CustomException
from src.data_preprocessing import prepare_datasets
from src.evaluate import compute_metrics


MODELS_DIR = BASE_DIR / "models" / "ml"


def train_models():
    try:
        logger.info("=" * 60)
        logger.info("  GeoMind — ML Training Pipeline (weather+temporal only)")
        logger.info("=" * 60)

        X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_datasets()
        logger.info(f"Feature count: {len(feat_names)}")

        # ── Model definitions ───────────────────────────────────────────
        # Hyperparameters chosen to balance accuracy vs. training time
        # for the ~33k-row training set without lag features.
        models = {
            "ridge_regression": Ridge(
                alpha=10.0,
                random_state=42
            ),

            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),

            "xgboost": XGBRegressor(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=5,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric="rmse",
                early_stopping_rounds=30,
                verbosity=0,
            ),
        }

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        results = []

        for name, model in models.items():
            logger.info(f"\nTraining: {name}")
            start = time.time()

            # XGBoost supports early stopping with eval_set
            if name == "xgboost":
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)

            elapsed = round(time.time() - start, 2)

            val_metrics  = compute_metrics(y_val,  model.predict(X_val))
            test_metrics = compute_metrics(y_test, model.predict(X_test))

            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved → {model_path}")

            results.append({
                "Model":      name,
                "Val MAE":    val_metrics["mae"],
                "Val RMSE":   val_metrics["rmse"],
                "Val R2":     val_metrics["r2"],
                "Test MAE":   test_metrics["mae"],
                "Test RMSE":  test_metrics["rmse"],
                "Test R2":    test_metrics["r2"],
                "Train Time": f"{elapsed}s",
            })

            print(
                f"[{name}] "
                f"Val MAE={val_metrics['mae']:,.0f} | "
                f"Val R2={val_metrics['r2']:.4f} | "
                f"Test MAE={test_metrics['mae']:,.0f} | "
                f"Test R2={test_metrics['r2']:.4f} | "
                f"{elapsed}s"
            )

        leaderboard = (
            pd.DataFrame(results)
            .sort_values("Val MAE")
            .reset_index(drop=True)
        )
        return leaderboard

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    leaderboard = train_models()
    print("\n" + "=" * 80)
    print("  GeoMind ML Leaderboard (sorted by Val MAE ↑)")
    print("=" * 80)
    print(leaderboard.to_string(index=False))
    print("=" * 80)
