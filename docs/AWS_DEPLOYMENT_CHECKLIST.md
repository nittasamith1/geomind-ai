# GeoMind — AWS EC2 Deployment Checklist

Use this pre-flight checklist to verify every component before and after deploying to an AWS EC2 instance.

---

## 1. Codebase & Version Control
- [x] **GitHub repository clean**: No unneeded binary cache, temp files, or virtual environment directories.
- [x] **No secrets committed**: No AWS credentials, access keys, or sensitive environment tokens in source code.
- [x] **`.gitignore` configured**: Protects `.venv/`, `__pycache__/`, `logs/*.log`, `.env`, while ensuring production `.joblib` model artifacts are tracked.
- [x] **`.env.example` available**: Clear template for configurable parameters (`ENVIRONMENT`, `HOST`, `PORT`, `LOG_LEVEL`, `CORS_ORIGINS`).
- [x] **No Windows absolute paths**: All paths dynamically constructed using `pathlib.Path` relative to `BASE_DIR`. Works seamlessly from any directory on Windows or Linux.

---

## 2. Models & Preprocessing Integrity
- [x] **Preprocessor available**: `models/preprocessor.joblib` exists (53 engineered features matching production inference).
- [x] **XGBoost model available**: `models/ml/xgboost.joblib` exists (~913 KB).
- [x] **Random Forest model available**: `models/ml/random_forest.joblib` compressed to ~39.9 MB (under GitHub's 100 MB hard limit).
- [x] **Ridge Regression model available**: `models/ml/ridge_regression.joblib` exists (< 1 KB).
- [x] **Zero retraining at startup**: Models and preprocessor loaded once during application startup lifespan.
- [x] **Zero target/lag leakage**: All models predict using only date/time and weather/holiday inputs; no current traffic volume needed.

---

## 3. Application Runtime & Dependencies
- [x] **Minimal requirements**: `requirements.txt` contains only the CPU runtime stack (NumPy, Pandas, Scikit-Learn, XGBoost, FastAPI, Uvicorn, Joblib, Pydantic).
- [x] **Dev dependencies separated**: `requirements-dev.txt` contains test runners (`pytest`, `httpx`).
- [x] **FastAPI starts without `--reload`**: Production server binds cleanly using `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- [x] **Binding interface**: Listens on `0.0.0.0` (all interfaces), not localhost, permitting external public access.

---

## 4. Endpoints & Frontend Verification
- [x] **Health check endpoint**: `GET /health` returns JSON with status, service, environment, and model loading status within milliseconds without invoking heavy inference.
- [x] **Interactive Swagger docs**: `GET /docs` serves interactive OpenAPI specification with request schemas and examples.
- [x] **Prediction endpoint**: `POST /predict` accepts single-observation payload and returns predicted volume, congestion index, level classification, and latency.
- [x] **Frontend static serving**: `GET /` serves responsive UI directly from FastAPI.
- [x] **Frontend dynamic API base**: `frontend/index.html` uses relative path for HTTP/HTTPS requests so it works on any EC2 public IP or custom domain without hard-coded `localhost`.
- [x] **Input validation & error handling**: Bad datetime, invalid temperature, or missing fields return clear HTTP 400/422 responses.

---

## 5. AWS EC2 Cloud Infrastructure
- [ ] **Instance provisioned**: Ubuntu 22.04 or 24.04 LTS instance running on AWS EC2 (`t2.micro` or `t3.micro`).
- [ ] **Security Group configured**:
  - [ ] Port `22` (SSH) open to your IP address.
  - [ ] Port `8000` (FastAPI) open to `0.0.0.0/0` (Anywhere IPv4).
  - [ ] Port `80` (HTTP) open if using Nginx reverse proxy.
- [ ] **Virtual environment active**: Python virtual environment (`.venv`) created and dependencies installed.
- [ ] **Test execution on EC2**: `python tests/run_tests.py` passes with 100% success on the Ubuntu server.

---

## 6. Process Daemon & Maintenance
- [ ] **Systemd service installed**: `deployment/geomind.service` copied to `/etc/systemd/system/geomind.service`.
- [ ] **Service enabled on boot**: `sudo systemctl enable geomind.service` executed.
- [ ] **Crash recovery verified**: Service automatically restarts if the Uvicorn process is terminated.
- [ ] **Logging verified**: `journalctl -u geomind.service -f` and `logs/YYYY_MM_DD.log` display structured startup and inference telemetry.
- [ ] **Reboot persistence tested**: Server reboot tested (`sudo reboot`) and application verified online upon restart.
- [ ] **Update procedure documented**: Git pull and service restart workflow verified.
