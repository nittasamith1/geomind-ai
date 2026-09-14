# GeoMind AI: Self-Learning Urban Traffic Forecasting & Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Targeted Role: **Amazon Applied Scientist I (Intern)**  
> Core Domain: Multivariate Time-Series Forecasting, Deep Sequential Architectures, ML Systems, Model Interpretability & Uncertainty.

---

## 1. Project Overview & Research Motivation
Urban traffic dynamics exhibit strong spatio-temporal autocorrelations, diurnal and weekly seasonalities, and stochastic shocks driven by severe weather and holidays.

**GeoMind AI** formulates traffic volume prediction as a contextual multivariate time-series problem:
$$\hat{y}_{t+h} = f\left(\{y_{t-k}, \mathbf{x}_{t-k}\}_{k=0}^{K}, \mathbf{z}_{t+h}\right)$$

Where $y$ represents traffic volume, $\mathbf{x}$ are exogenous weather telemetry features, $\mathbf{z}$ are deterministic calendar features, and $h$ is the prediction horizon.

### Key Questions Investigated:
1. **Autoregressive vs. Exogenous Value:** How much variance is captured by historical traffic lags vs. meteorological signals?
2. **Tabular GBDTs vs. Sequential Deep Learning:** Under what sample sizes and temporal horizons do sequence models (LSTM / GRU) outperform gradient boosted decision trees (XGBoost / LightGBM)?
3. **Error Anatomy & Tail Risk:** In which regime (rush-hour transition, extreme weather, public holidays) do models incur catastrophic residual spikes?
4. **Autonomous Model Drift & Self-Learning:** How can the system continuously evaluate new telemetry and trigger retraining only when a statistically significant gain is demonstrated?

---

## 2. Architecture & Directory Structure

```text
GeoMind-AI/
│
├── data/
│   ├── raw/                  # Real raw sensor telemetry (48,204 rows)
│   └── processed/            # Deduplicated & sanitized train/val/test splits
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis & statistical tests
│   ├── 02_feature_engineering.ipynb # Feature transformations & cyclical encodings
│   ├── 03_ml_models.ipynb    # Baselines, Linear, Random Forest, XGBoost
│   ├── 04_deep_learning.ipynb# LSTM & GRU PyTorch sequence models
│   └── 05_model_evaluation.ipynb # Error analysis, SHAP, & uncertainty intervals
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py     # Ingestion, timestamp deduplication, & split
│   ├── data_preprocessing.py # Scalers, encoders, and leakage-safe transforms
│   ├── feature_engineering.py# Lags, rolling windows, & cyclical features
│   ├── train_ml.py           # ML model training & hyperparameter search
│   ├── train_dl.py           # PyTorch sequence dataset & training loops
│   ├── evaluate.py           # MAE, RMSE, R2, and regime error analysis
│   └── predict.py            # Unified inference engine for ML & DL models
│
├── models/
│   ├── ml/                   # Serialized ML models (joblib)
│   └── dl/                   # Serialized PyTorch checkpoints (.pt)
│
├── experiments/
│   └── results.csv           # Experiment tracking & model registry
│
├── api/
│   ├── main.py               # FastAPI application with lifecycle management
│   ├── schemas.py            # Pydantic request & response contracts
│   └── prediction.py         # Serving routes for single & multi-step forecast
│
├── tests/
│   ├── test_preprocessing.py # Ingestion & split non-leakage tests
│   ├── test_features.py      # Feature transformation & cyclical integrity
│   └── test_prediction.py    # API contracts & inference robustness
│
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## 3. Dataset Characteristics & Integrity Findings
- **Data Source:** Metro Interstate Traffic Volume (I-94 Westbound between Minneapolis and St. Paul, MN; MnDOT / UCI).
- **Time Span:** 2012-10-02 to 2018-09-30 (6 years).
- **Target Variable:** `traffic_volume` (Hourly vehicle count, mean: 3,260, max: 7,280).
- **Timestamp Duplicates:** 7,629 duplicate hours resolved via scientific aggregation.
- **Sensor Anomalies:** 10 records with unphysical $0.0\text{ K}$ (-273.15 °C) temperature and 1 record with $9,831.3\text{ mm}$ rain spike sanitized via time-weighted interpolation and boundary clipping.
- **Chronological Split:** Train (70% = 28,402 hrs), Validation (15% = 6,086 hrs), Test (15% = 6,087 hrs) with mathematical verification of zero future leakage.

---

## 4. Experimental Results & Leaderboard

All models evaluated strictly on the out-of-time test set (6,087 sequential hourly intervals). Experiments are tracked in `experiments/results.csv`.

| Model Family | Architecture / Algorithm | Sequence Length ($L$) | Test MAE (veh/hr) | Test RMSE (veh/hr) | Test $R^2$ | Inference Time | Status |
|---|---|---|---|---|---|---|---|
| **Baseline** | Historical Mean | — | 1,745.2 | 1,986.4 | 0.000 | < 1 ms | Reference |
| **Baseline** | Persistence ($y_t$) | — | 412.8 | 684.3 | 0.881 | < 1 ms | Reference |
| **ML** | Ridge Regression | — | 289.4 | 451.2 | 0.948 | 2 ms | Baseline |
| **ML** | Random Forest (150 trees) | — | 148.6 | 291.5 | 0.978 | 138 ms | Benchmark |
| **ML (Champion)** | **XGBoost (Histogram-based)** | — | **130.0** | **272.4** | **0.981** | **103 ms** | **Production Champion** |
| **Deep Learning** | PyTorch LSTM (2-layer + Huber) | $L=6$ hrs | **212.5** | **346.7** | **0.969** | 71 ms | DL Champion |
| **Deep Learning** | PyTorch GRU (2-layer + Huber) | $L=12$ hrs | 223.7 | 339.4 | 0.971 | 100 ms | Efficient DL |
| **Deep Learning** | PyTorch LSTM (2-layer + Huber) | $L=24$ hrs | 222.8 | 348.3 | 0.969 | 85 ms | Long Horizon |
| **Deep Learning** | PyTorch LSTM (2-layer + Huber) | $L=12$ hrs | 228.1 | 368.9 | 0.965 | 78 ms | Comparison |

---

## 5. Applied Scientist Insights & Research Findings

### 1. Tabular GBDT vs. Deep Sequence Modeling
- **XGBoost achieves the lowest test error (MAE: 130.0 vs. LSTM's 212.5)** when provided with domain-engineered temporal lags ($t-1, t-2, t-3, t-24$) and rolling window statistics ($3\text{h}, 6\text{h}, 24\text{h}$).
- The inductive bias of gradient boosted trees on engineered tabular features excels because traffic exhibits rigid periodicities (diurnal rush hours, weekly cycles) that explicit lags capture directly without requiring sequence optimization.
- **Deep sequential models (LSTM/GRU)** shine in raw sequence spaces without handcrafted features, achieving strong $R^2 > 0.969$ with fast sub-100ms inference.

### 2. Context Length Ablation ($L \in \{6, 12, 24\}$)
- **Short context ($L=6$) outperforms $L=12$ and $L=24$ for LSTM**: In next-hour forecasting, recent short-term momentum dominates. Longer unrolled sequences introduce slight gradient noise and overfitting on non-stationary regimes.
- **GRU vs. LSTM**: GRU achieves competitive performance (MAE 223.7 vs 228.1 at $L=12$) with 25% fewer gating parameters and faster convergence.

### 3. Error Slicing & Residual Analysis
- **Rush-Hour Transitions (7–9 AM, 4–6 PM)** account for the largest variance in residual errors due to sudden demand influx and stochastic congestion onset.
- **Precipitation Regimes**: Extreme rain and snow produce systematic under-predictions if models rely solely on calendar features; exogenous weather signals act as a crucial dampener on predicted speed/capacity.
- **SHAP TreeExplainer**: Quantified that `traffic_lag_1` (previous hour volume) contributes 42% of total feature attribution, followed by `hour_sin` / `hour_cos` (26%), and rolling 3-hour mean (14%).

---

## 6. Production Model Serving (FastAPI)

Production-ready, asynchronous REST API serving both traditional ML and PyTorch Deep Learning models.

### API Capabilities:
- **Lifespan Startup:** Eagerly warms and caches all model checkpoints in memory (zero cold-start latency).
- **Strict Contract Validation:** Pydantic models validate sensor limits (temperature 200–330K, non-negative traffic volume, ISO-8601 timestamps).
- **Dual Inference Paradigms:**
  - Single-observation forecast via contextual temporal padding.
  - Multi-observation batch forecast with rolling temporal context.
- **Production Metrics:** Every response includes model metadata, forecast horizon, and measured execution time ($< 150\text{ ms}$).

### Quick Start:

```bash
# 1. Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 2. Interactive Swagger UI
open http://localhost:8000/docs
```

#### Example Single Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "date_time": "2024-08-15T08:00:00",
      "temp": 287.5,
      "rain_1h": 0.0,
      "snow_1h": 0.0,
      "clouds_all": 40.0,
      "weather_main": "Clear",
      "holiday": "None",
      "traffic_volume": 4200.0
    },
    "model_type": "xgboost"
  }'
```

---

## 7. Verification & Testing

The system is tested end-to-end with **65 automated unit and integration tests** (100% pass rate):

```bash
# Run all unit and integration tests
pytest tests/ -v
```

- `test_features.py` (25 tests): Verifies temporal leakage prevention, cyclical trigonometry identities, rolling-window bounds, and NaN handling.
- `test_preprocessing.py` (10 tests): Verifies train-only scaler fitting, serialization round-trips, and dimension consistency.
- `test_prediction.py` (30 tests): Verifies FastAPI routes, input boundary validation, batch processing, response schemas, and inference latency.

---

## 8. Alignment with Amazon Applied Scientist I Competencies

| Amazon Applied Scientist Expectation | Demonstrated in GeoMind AI |
|---|---|
| **Problem Formulation** | Formulated non-stationary urban traffic as autoregressive multivariate time-series forecasting with exogenous features. |
| **Statistical & ML Rigor** | Temporal train/val/test splits without leakage; strong baselines (Mean, Persistence); ablation studies ($L=6, 12, 24$). |
| **Deep Learning** | PyTorch sequential models (LSTM, GRU), Huber loss optimization, early stopping, target normalization. |
| **Model Interpretability** | SHAP TreeExplainer global and local feature attributions; sliced residual error analysis across time and weather regimes. |
| **Production Engineering** | Clean modular architecture (`src/`, `api/`, `tests/`), FastAPI microservice, Pydantic schemas, 65 automated tests. |

