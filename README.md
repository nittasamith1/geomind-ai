# 🚦 GeoMind — AI Urban Traffic Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

> **GeoMind** is an end-to-end Classical Machine Learning project designed to forecast urban highway traffic volume ($R^2 > 0.957$) using purely meteorological and temporal signals. 
> 
> **Zero Prior Traffic Volume Required:** Unlike naive autoregressive approaches that require the user to already know current traffic counts, GeoMind predicts traffic from date/time and weather conditions alone.

---

## 🌟 Key Highlights

- 🎯 **High Accuracy ($R^2 = 0.957$ / $\text{MAE} \approx 243\text{ veh/hr}$):** SOTA performance among classical ML models on the Metro Interstate Traffic Volume dataset.
- 🚫 **No Present Traffic Input Needed:** Eliminates the unrealistic assumption that end users know real-time sensor traffic counts.
- ⚡ **Ultra-Fast Inference (< 2 ms):** Lightweight classical ML footprint runs seamlessly on any CPU without GPU dependencies.
- 🤖 **Multi-Model Selector:** Compare predictions across **XGBoost**, **Random Forest**, and **Ridge Regression** directly in the UI.
- 🕐 **Automatic Date & Time Detection:** Frontend auto-fetches system date/time with click-to-edit flexibility.
- 🎨 **Modern Web Interface:** Dark glassmorphic design featuring animated counter metrics, traffic congestion indicators, and real-time Kelvin-to-Celsius conversions.

---

## 📊 Model Leaderboard

All models evaluated on chronological train ($70\%$), validation ($15\%$), and test ($15\%$) partitions of the Metro Interstate dataset (~48,000 hourly observations):

| Rank | Model | Paradigm | Val MAE | Val $R^2$ | Test MAE | Test $R^2$ | Inference | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Regressor** | Gradient Boosted Trees | **273.21** | **0.9553** | **243.22** | **0.9570** | **~1.5 ms** | **Best** |
| 🥈 | **Random Forest** | Bagged Decision Trees | 281.27 | 0.9522 | 251.86 | 0.9557 | ~2.2 ms | Candidate |
| 🥉 | **Ridge Regression** | Regularized Linear ($L_2$) | 796.56 | 0.7319 | 791.73 | 0.7450 | < 1 ms | Baseline |
| — | **Unconditional Mean**| Statistical Baseline | 1,723.00 | -0.0044 | 1,735.79 | -0.0033 | 0 ms | Lower Bound |

*Units: MAE in vehicles/hour. Traffic volume spans 0 to 7,280 veh/hr.*

---

## 📐 System Architecture

```
[ User Browser / Client ]
           │
           ▼
[ FastAPI Application (api/main.py) ]
    ├── Static Mount: /static (frontend/index.html)
    ├── Documentation: /docs (OpenAPI / Swagger)
    └── POST /predict
           │
           ▼
[ Inference Pipeline (api/prediction.py) ]
    │
    ├── 1. Build Single Observation Vector
    ├── 2. Feature Engineering (src/feature_engineering.py)
    │       ├── Cyclical Time Encodings (sin/cos of hour, DOW, month, DOY)
    │       ├── Commuter Regime Buckets (morning & evening rush, midday, night)
    │       ├── Weather Friction & Comfort Indices (Celsius, freeze, heavy rain/snow)
    │       └── Non-linear Cross-interactions
    │
    ├── 3. Preprocessing Transformation (models/preprocessor.joblib)
    │       ├── StandardScaler (numeric features)
    │       └── OneHotEncoder (weather category)
    │
    └── 4. Model Scoring (models/ml/*.joblib)
            └── Returns Predicted Volume (veh/hr) + Congestion Level + Latency
```

---

## 📥 Prediction Input Specification

The API and Web Interface accept only inputs that an end user or automated weather API can provide:

| Input Field | Type | Unit / Format | Description & Example |
| :--- | :---: | :---: | :--- |
| `date_time` | String | ISO 8601 | Auto-filled timestamp (e.g. `2026-10-01T08:00:00`) |
| `model_type` | String | Categorical | `xgboost` (default), `random_forest`, or `ridge_regression` |
| `temp` | Float | Kelvin | Ambient temperature (e.g. `288.15` K $\approx 15^\circ$C) |
| `weather_main`| String | Categorical | `Clear`, `Clouds`, `Rain`, `Snow`, `Mist`, `Thunderstorm`, etc. |
| `rain_1h` | Float | mm | Rainfall accumulation in the last hour (e.g. `0.0`) |
| `snow_1h` | Float | mm | Snowfall accumulation in the last hour (e.g. `0.0`) |
| `clouds_all` | Integer | % | Cloud cover percentage (`0` to `100`%) |
| `holiday` | String | Categorical | US Federal/State holiday name, or `'None'` |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ installed.

```powershell
# Clone the repository
git clone https://github.com/nittasamith1/geomind-ai.git
cd GeoMind

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
Ingests data, executes feature engineering, fits transformers, and trains all models:

```powershell
python run_pipeline.py
```

### 3. Launch the Web Application & API

#### Local Windows Development:
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- 🌐 **Web Interface:** [http://localhost:8000](http://localhost:8000)
- 📖 **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- 🩺 **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## ☁️ AWS EC2 Deployment

GeoMind is fully configured for direct deployment on an **AWS EC2 Ubuntu (22.04 / 24.04 LTS)** CPU instance (e.g. `t2.micro` Free Tier) without Docker or cloud lock-in.

### Quick Start on Ubuntu EC2:

```bash
# 1. Update system packages
sudo apt update && sudo apt install python3 python3-pip python3-venv git -y

# 2. Clone repository & enter directory
git clone https://github.com/<your-username>/GeoMind.git
cd GeoMind

# 3. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify models & preprocessing
python tests/run_tests.py

# 6. Start production server (bound to 0.0.0.0)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Production Background Service (`systemd`):
```bash
sudo cp deployment/geomind.service /etc/systemd/system/geomind.service
sudo systemctl daemon-reload
sudo systemctl enable geomind.service
sudo systemctl start geomind.service
sudo systemctl status geomind.service
```

### Public AWS Deployment URLs:
- 🌐 **Web UI:** `http://<EC2_PUBLIC_IP>:8000/`
- 📖 **Swagger API Docs:** `http://<EC2_PUBLIC_IP>:8000/docs`
- 🩺 **Health Check:** `http://<EC2_PUBLIC_IP>:8000/health`
- 🔮 **Inference Endpoint:** `http://<EC2_PUBLIC_IP>:8000/predict`

*(For full step-by-step instructions from creating EC2 security groups to viewing journalctl logs, see [docs/AWS_EC2_DEPLOYMENT.md](docs/AWS_EC2_DEPLOYMENT.md) and the [docs/AWS_DEPLOYMENT_CHECKLIST.md](docs/AWS_DEPLOYMENT_CHECKLIST.md).)*

---

## 🧪 Verification & Testing

Execute the automated verification suite to test model serialization, inference integrity, and regime contrasts:

```powershell
python tests/run_tests.py
```

Expected output:
```text
Testing model loading...
Status: {'preprocessor': True, 'xgboost': True, 'random_forest': True, 'ridge_regression': True}

Running inference across all models:
  Model: xgboost            | Prediction: 5617.0 veh/hr | Level: Very High
  Model: random_forest      | Prediction: 5730.0 veh/hr | Level: Very High
  Model: ridge_regression   | Prediction: 4195.4 veh/hr | Level: High

Comparing Rush hour (8 AM: 5617.0 veh/hr) vs Night (2 AM: 356.6 veh/hr):
  Comparison validation passed: rush hour traffic is significantly higher than night traffic!
Storm weather traffic prediction at 8 AM: 5446.8 veh/hr (High)

ALL VERIFICATIONS PASSED SUCCESSFULLY!
```

---

## 📡 API Reference

### `POST /predict`
Predict traffic volume for a target hour and weather condition.

**Request Body:**
```json
{
  "observation": {
    "date_time": "2026-10-01T08:00:00",
    "temp": 288.15,
    "rain_1h": 0.0,
    "snow_1h": 0.0,
    "clouds_all": 20.0,
    "weather_main": "Clear",
    "holiday": "None"
  },
  "model_type": "xgboost"
}
```

**Response Body (200 OK):**
```json
{
  "status": "success",
  "model_used": "xgboost",
  "input_datetime": "2026-10-01T08:00:00",
  "predicted_traffic_volume": 5617.0,
  "traffic_level": "Very High",
  "unit": "vehicles/hr",
  "inference_ms": 1.48
}
```

---

## 📁 Repository Structure

```
GeoMind/
├── api/
│   ├── main.py                 # FastAPI application & route declarations
│   ├── prediction.py           # Model loading & inference engine
│   └── schemas.py              # Pydantic request & response contracts
├── data/
│   ├── raw/                    # Metro Interstate Traffic Volume dataset
│   └── processed/              # train.csv, val.csv, test.csv
├── docs/                       # Comprehensive scientific documentation
│   ├── 01_problem.md           # Problem formulation & mathematical objectives
│   ├── 02_dataset.md           # Dataset provenance & schema profiling
│   ├── 03_eda.md               # Exploratory data analysis & empirical patterns
│   ├── 04_feature_engineering.md # Contextual & meteorological feature rationale
│   ├── 05_baseline_experiments.md# Statistical baselines benchmark
│   └── 06_ml_experiments.md    # Classical ML leaderboard & trade-off analysis
├── frontend/
│   └── index.html              # Modern glassmorphism web UI
├── models/
│   ├── preprocessor.joblib     # Fitted StandardScaler & OneHotEncoder
│   └── ml/                     # Serialized classical ML models
│       ├── xgboost.joblib
│       ├── random_forest.joblib
│       └── ridge_regression.joblib
├── src/
│   ├── data_ingestion.py       # Cleaning, deduplication & splitting
│   ├── data_preprocessing.py   # Transformer pipelines
│   ├── evaluate.py             # MAE, RMSE, R² metrics
│   ├── exception.py            # Traceback exception handling
│   ├── feature_engineering.py  # Pure contextual & meteorological features
│   ├── logger.py               # Centralized logging configuration
│   └── train_ml.py             # Model training & leaderboard generation
├── tests/
│   ├── run_tests.py            # Standalone end-to-end verification script
│   ├── test_features.py        # Feature engineering tests
│   ├── test_prediction.py      # FastAPI prediction route tests
│   └── test_preprocessing.py   # Transformer pipeline tests
├── requirements.txt            # Project dependencies
├── run_pipeline.py             # One-click end-to-end pipeline runner
└── README.md                   # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
