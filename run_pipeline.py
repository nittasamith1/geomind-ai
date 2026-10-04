"""
Run this script to train all ML models from scratch.

Steps:
  1. Ingests and cleans raw data
  2. Splits into train / val / test (70/15/15)
  3. Engineers features (lags, rolling stats, time features)
  4. Fits preprocessor and saves to models/preprocessor.joblib
  5. Trains Ridge Regression, Random Forest, XGBoost
  6. Saves each model to models/ml/
  7. Prints final leaderboard
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_ingestion import run_data_ingestion
from src.train_ml import train_models


if __name__ == "__main__":
    print("=" * 60)
    print("  Step 1: Data Ingestion")
    print("=" * 60)
    run_data_ingestion()

    print("\n" + "=" * 60)
    print("  Step 2: Model Training")
    print("=" * 60)
    leaderboard = train_models()

    print("\n" + "=" * 60)
    print("  GeoMind ML Leaderboard (sorted by Val MAE)")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    print("=" * 60)
    print("\nDone! Models saved to models/ml/")
    print("Run the API: python -m uvicorn api.main:app --reload --port 8000")
