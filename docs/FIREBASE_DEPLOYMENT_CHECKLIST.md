# GeoMind — Firebase Deployment Checklist

Use this pre-flight checklist to verify every component before and after deploying to Firebase + Google Cloud Run.

---

## 1. Codebase & Version Control

- [x] **GitHub repository clean**: No unneeded binary cache, temp files, or virtual environment directories.
- [x] **No secrets committed**: No Firebase service account keys, GCP credentials, or access tokens in source code.
- [x] **`.gitignore` configured**: Protects `.venv/`, `__pycache__/`, `logs/*.log`, `.env`, `firebase-adminsdk*.json`, `.firebase/`. Ensures production `.joblib` model artifacts are tracked.
- [x] **`.env.example` available**: Clear template for `ENVIRONMENT`, `HOST`, `PORT` (8080), `GCP_PROJECT_ID`, `FIREBASE_PROJECT_ID`, `FIRESTORE_COLLECTION`, `CORS_ORIGINS`.
- [x] **No Windows absolute paths**: All paths dynamically constructed using `pathlib.Path` relative to `BASE_DIR`.
- [x] **Port**: Application binds to `PORT` env var (default 8080) — Cloud Run injects this automatically.

---

## 2. Models & Preprocessing Integrity

- [x] **Preprocessor available**: `models/preprocessor.joblib` exists (53 engineered features matching production inference).
- [x] **XGBoost model available**: `models/ml/xgboost.joblib` exists (~913 KB).
- [x] **Random Forest model available**: `models/ml/random_forest.joblib` compressed to ~39.9 MB (under GitHub's 100 MB hard limit).
- [x] **Ridge Regression model available**: `models/ml/ridge_regression.joblib` exists (< 1 KB).
- [x] **Zero retraining at startup**: Models and preprocessor loaded once during application startup lifespan.
- [x] **Zero target/lag leakage**: All models predict using only date/time and weather/holiday inputs; no current traffic volume needed.
- [x] **Models included in Docker image**: `.dockerignore` does NOT exclude `models/` — they must be baked into the container.

---

## 3. Firebase & Google Cloud Configuration

- [ ] **Firebase project created**: Project exists in [Firebase Console](https://console.firebase.google.com/).
- [ ] **`.firebaserc` updated**: Contains correct project ID (not the default `geomind-ai` placeholder).
- [ ] **Cloud Firestore enabled**: Firestore database created in Firebase Console.
- [ ] **Cloud Run API enabled**: `run.googleapis.com` enabled in GCP project.
- [ ] **Cloud Build API enabled**: `cloudbuild.googleapis.com` enabled.
- [ ] **firebase.json correct**: Rewrites `/api/**` to Cloud Run service; static files from `frontend/`.

---

## 4. Docker & Cloud Run

- [ ] **Docker image builds successfully**: `docker build .` completes without errors locally.
- [ ] **`.dockerignore` correct**: Excludes `.venv/`, tests, docs, `data/`, notebooks, but **includes** `models/`.
- [ ] **Cloud Build succeeds**: `gcloud builds submit` completes and image pushed to `gcr.io`.
- [ ] **Cloud Run revision deployed**: Service `geomind-api` is live in the specified region.
- [ ] **Cloud Run allows unauthenticated access**: `--allow-unauthenticated` flag used (required for public API).
- [ ] **Memory set to 1Gi**: Random Forest model requires at least 512Mi; 1Gi ensures stability.
- [ ] **CORS_ORIGINS env var set**: Firebase Hosting URLs added to Cloud Run environment variables.

---

## 5. Firebase Hosting

- [ ] **Firebase CLI authenticated**: `firebase login` successful.
- [ ] **Hosting initialized**: `firebase.json` present with correct `public: "frontend"` directory.
- [ ] **Rewrite rule configured**: `/api/**` proxies to `geomind-api` Cloud Run service.
- [ ] **Frontend deployed**: `firebase deploy --only hosting` succeeds.
- [ ] **Hosting URL live**: `https://YOUR_PROJECT_ID.web.app` serves the GeoMind SPA.
- [ ] **No hardcoded localhost**: `frontend/index.html` uses relative API paths (works on any domain).

---

## 6. Endpoints & Integration Verification

- [ ] **Health check endpoint**: `GET /health` returns `{"status": "healthy", "api_version": "4.0.0"}`.
- [ ] **Swagger docs accessible**: `GET /docs` serves interactive OpenAPI specification.
- [ ] **Prediction endpoint works**: `POST /predict` returns predicted volume, level, congestion, and latency.
- [ ] **Firestore logging active**: After a prediction, a new document appears in Firestore `predictions` collection.
- [ ] **Frontend → API connection**: Browser on Firebase Hosting URL successfully calls Cloud Run `/predict`.
- [ ] **Input validation works**: Bad datetime or invalid temperature returns HTTP 400 with clear message.

---

## 7. Firestore Data Logging

- [ ] **Firestore client initialises**: No errors in Cloud Run logs about Firebase Admin SDK.
- [ ] **Prediction documents created**: Documents appear in `predictions` collection after inference.
- [ ] **Service account has Firestore access**: Cloud Run service account has `roles/datastore.user` IAM role.
- [ ] **Firestore security rules configured**: Rules set appropriately (write from backend only, read locked).

---

## 8. Cost & Performance

- [ ] **Min instances set to 0**: `--min-instances 0` for cost savings on low-traffic portfolio projects.
- [ ] **Cold start acceptable**: First request may take 5–10 seconds (model loading). Warm requests < 2ms.
- [ ] **Free tier confirmed**: Cloud Run, Firebase Hosting, and Firestore usage within free tier limits.
- [ ] **No accidental billing**: No large-scale data ingestion or continuous heavy requests running.

---

## 9. Cleanup Checklist (When Done)

- [ ] Cloud Run service deleted: `gcloud run services delete geomind-api --region us-central1`
- [ ] Container images deleted: `gcloud container images delete gcr.io/PROJECT_ID/geomind-api`
- [ ] Firebase Hosting disabled: `firebase hosting:disable`
- [ ] Firestore data cleared (if needed): Delete collections from Firebase Console
