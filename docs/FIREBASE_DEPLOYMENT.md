# GeoMind — Firebase + Cloud Run Deployment Guide

This guide provides step-by-step instructions for deploying **GeoMind AI Urban Traffic Forecasting** on **Firebase Hosting** (frontend) and **Google Cloud Run** (FastAPI backend), with **Cloud Firestore** for prediction logging.

---

## Architecture Overview

```
                                  ┌──────────────────────────────────────┐
                                  │         Firebase Hosting              │
 ┌─────────────────┐              │   https://geomind-ai.web.app          │
 │  Client Browser │ ─────────►  │   frontend/index.html  (SPA)          │
 └─────────────────┘              │                                       │
                                  │  Rewrites: /api/** ──────────────┐    │
                                  └──────────────────────────────────│────┘
                                                                      │
                                  ┌───────────────────────────────────▼──┐
                                  │         Google Cloud Run              │
                                  │   geomind-api  (Docker + Uvicorn)     │
                                  │                                        │
                                  │   GET  /health                         │
                                  │   GET  /docs  (OpenAPI)                │
                                  │   GET  /models                         │
                                  │   POST /predict ──────────────────┐    │
                                  └───────────────────────────────────│────┘
                                                                       │
                                  ┌────────────────────────────────────▼──┐
                                  │         Cloud Firestore               │
                                  │   Collection: predictions              │
                                  │   (prediction logs & analytics)        │
                                  └────────────────────────────────────────┘
```

---

## 1. Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Google Cloud SDK** (`gcloud`) | Latest | https://cloud.google.com/sdk/docs/install |
| **Firebase CLI** | Latest | `npm install -g firebase-tools` |
| **Docker Desktop** | Latest | https://docs.docker.com/get-docker/ |
| **Python** | 3.10+ | https://python.org |

---

## 2. Create a Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/).
2. Click **Add project** → name it `geomind-ai` (or your preferred name).
3. Enable **Google Analytics** (optional).
4. Click **Create project**.

> **Important**: Note your **Project ID** (shown in Project Settings). You'll need it throughout this guide.

---

## 3. Enable Google Cloud APIs

In the [Google Cloud Console](https://console.cloud.google.com/), select your project and enable:

```bash
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    firestore.googleapis.com \
    firebase.googleapis.com \
    --project=YOUR_PROJECT_ID
```

---

## 4. Initialise Cloud Firestore

1. In Firebase Console → **Firestore Database** → **Create database**.
2. Choose **Production mode** (or Start in test mode for development).
3. Select a location (e.g., `nam5` for US, `eur3` for Europe).
4. Click **Enable**.

The backend will automatically create the `predictions` collection on the first inference call.

---

## 5. Configure the Firebase Project Locally

```bash
# Clone the repository
git clone https://github.com/nittasamith1/geomind-ai.git
cd GeoMind

# Log in to Firebase
firebase login

# Set the active Firebase project
firebase use geomind-ai   # ← replace with your project ID
```

Verify `.firebaserc` contains your project:
```json
{
  "projects": {
    "default": "geomind-ai"
  }
}
```

---

## 6. Local Development Setup

```powershell
# Windows: Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install all dependencies (including Firebase Admin SDK)
pip install -r requirements.txt

# Copy and edit environment configuration
copy .env.example .env
# Edit .env — set GCP_PROJECT_ID=your-project-id
```

### Run the ML Pipeline (trains models locally):

```powershell
python run_pipeline.py
```

### Start the API locally:

```powershell
# Set PORT for local (Cloud Run uses 8080 by default)
$env:PORT = "8080"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

- 🌐 **Web Interface:** http://localhost:8080
- 📖 **Swagger Docs:** http://localhost:8080/docs
- 🩺 **Health Check:** http://localhost:8080/health

---

## 7. Deploy the FastAPI Backend to Cloud Run

### Option A: Automated Script (Recommended)

```bash
export GCP_PROJECT_ID=geomind-ai
chmod +x deployment/deploy_firebase.sh
./deployment/deploy_firebase.sh
```

### Option B: Manual Step-by-Step

**Step 1 — Authenticate:**
```bash
gcloud auth login
gcloud config set project geomind-ai
gcloud auth configure-docker
```

**Step 2 — Build and push the Docker image using Cloud Build:**
```bash
gcloud builds submit \
    --tag gcr.io/geomind-ai/geomind-api:latest \
    --project geomind-ai \
    .
```

**Step 3 — Deploy to Cloud Run:**
```bash
gcloud run deploy geomind-api \
    --image gcr.io/geomind-ai/geomind-api:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "ENVIRONMENT=production,GCP_PROJECT_ID=geomind-ai" \
    --project geomind-ai
```

**Step 4 — Get the Cloud Run URL:**
```bash
gcloud run services describe geomind-api \
    --platform managed \
    --region us-central1 \
    --format="value(status.url)"
```
You'll get a URL like: `https://geomind-api-xxxxxxxx-uc.a.run.app`

---

## 8. Deploy the Frontend to Firebase Hosting

Firebase Hosting serves the `frontend/index.html` SPA and proxies `/api/**` requests to Cloud Run automatically via `firebase.json` rewrites.

```bash
firebase deploy --only hosting --project geomind-ai
```

Expected output:
```text
=== Deploying to 'geomind-ai'...

i  deploying hosting
i  hosting[geomind-ai]: beginning deploy...
✔  hosting[geomind-ai]: file upload complete
i  hosting[geomind-ai]: finalizing version...
✔  hosting[geomind-ai]: version finalized
✔  hosting[geomind-ai]: release complete

✔  Deploy complete!

Project Console: https://console.firebase.google.com/project/geomind-ai/overview
Hosting URL: https://geomind-ai.web.app
```

---

## 9. Verify the Deployment

### Health Check
```bash
curl https://geomind-api-xxxxxxxx-uc.a.run.app/health
```
Expected response:
```json
{
  "status": "healthy",
  "service": "GeoMind",
  "environment": "production",
  "models_loaded": {
    "preprocessor": true,
    "xgboost": true,
    "random_forest": true,
    "ridge_regression": true
  },
  "api_version": "4.0.0"
}
```

### Inference Test
```bash
curl -X POST "https://geomind-api-xxxxxxxx-uc.a.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Public Access URLs
| Endpoint | URL |
|----------|-----|
| 🌐 **Web Dashboard** | `https://geomind-ai.web.app` |
| 🔮 **Predict API** | `https://geomind-api-*.run.app/predict` |
| 📖 **Swagger Docs** | `https://geomind-api-*.run.app/docs` |
| 🩺 **Health Check** | `https://geomind-api-*.run.app/health` |

---

## 10. View Prediction Logs in Firestore

1. Go to [Firebase Console](https://console.firebase.google.com/) → your project → **Firestore Database**.
2. Open the `predictions` collection.
3. Each document contains:
   - `model_name` — which ML model was used
   - `input_datetime` — the requested forecast time
   - `volume` — predicted traffic volume (veh/hr)
   - `level` — traffic level label
   - `congestion_pct` — road capacity percentage
   - `inference_ms` — model latency
   - `timestamp` — UTC server timestamp
   - `obs` — raw weather/holiday inputs

---

## 11. Updating After Code Changes

```bash
# 1. Make your changes locally and commit
git add .
git commit -m "feat: your update"
git push origin main

# 2. Rebuild and redeploy the Cloud Run backend
gcloud builds submit \
    --tag gcr.io/geomind-ai/geomind-api:latest .
gcloud run deploy geomind-api \
    --image gcr.io/geomind-ai/geomind-api:latest \
    --region us-central1 \
    --platform managed

# 3. Redeploy frontend (if index.html changed)
firebase deploy --only hosting
```

---

## 12. Troubleshooting

### ❌ Cloud Run container fails to start
- Check logs: `gcloud run services logs read geomind-api --region us-central1`
- Common cause: `models/ml/*.joblib` not in the image — ensure `.dockerignore` does not exclude `models/`

### ❌ CORS errors in browser
- Ensure your Firebase Hosting URL is in the `CORS_ORIGINS` environment variable on Cloud Run:
  ```bash
  gcloud run services update geomind-api \
      --set-env-vars "CORS_ORIGINS=https://geomind-ai.web.app,https://geomind-ai.firebaseapp.com"
  ```

### ❌ Firestore permission denied
- Ensure the Cloud Run service account has the **Cloud Datastore User** IAM role:
  ```bash
  gcloud projects add-iam-policy-binding geomind-ai \
      --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
      --role="roles/datastore.user"
  ```

### ❌ Firebase Hosting shows old version
```bash
firebase hosting:disable   # disable & re-enable
firebase deploy --only hosting
```

---

## 13. Cost Estimate (Free Tier)

| Service | Free Tier | Notes |
|---------|-----------|-------|
| **Cloud Run** | 2M requests/month | First 180K vCPU-seconds free |
| **Firebase Hosting** | 10 GB storage, 360 MB/day transfer | Generous free tier |
| **Firestore** | 50K reads, 20K writes/day | More than enough for demos |
| **Cloud Build** | 120 build-minutes/day | Sufficient for development |

GeoMind runs **entirely within the Google Cloud free tier** for typical demo/portfolio usage.

---

## 14. Clean Up Resources

To avoid any potential charges when your project is complete:

```bash
# Delete Cloud Run service
gcloud run services delete geomind-api --region us-central1 --quiet

# Delete container images
gcloud container images delete gcr.io/geomind-ai/geomind-api:latest --quiet

# Disable Firebase Hosting
firebase hosting:disable

# Or delete the entire Firebase project
# (Firebase Console → Project Settings → Delete project)
```
