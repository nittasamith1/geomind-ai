#!/usr/bin/env bash
# ==============================================================================
# GeoMind — Firebase + Cloud Run Deployment Script
# ==============================================================================
# Builds the Docker image, pushes it to Google Artifact Registry, deploys it to
# Cloud Run, and then deploys the frontend to Firebase Hosting.
#
# Prerequisites:
#   - Google Cloud SDK (gcloud) installed & authenticated
#   - Firebase CLI (npm i -g firebase-tools) installed & logged in
#   - Docker Desktop running
#
# Usage (from the project root):
#   chmod +x deployment/deploy_firebase.sh
#   ./deployment/deploy_firebase.sh
# ==============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-geomind-ai}"           # ← set GCP_PROJECT_ID env var or edit this
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="geomind-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
# ──────────────────────────────────────────────────────────────────────────────

echo "===================================================================="
echo "  GeoMind — Firebase + Cloud Run Deployment"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Service : ${SERVICE_NAME}"
echo "===================================================================="

# 1. Authenticate gcloud (skipped if already authenticated)
echo "[1/6] Verifying gcloud authentication..."
gcloud auth configure-docker --quiet
gcloud config set project "${PROJECT_ID}"

# 2. Enable required APIs (idempotent)
echo "[2/6] Enabling required Google Cloud APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    firestore.googleapis.com \
    firebase.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

# 3. Build and push Docker image using Cloud Build
echo "[3/6] Building and pushing Docker image via Cloud Build..."
gcloud builds submit \
    --tag "${IMAGE_NAME}:latest" \
    --project="${PROJECT_ID}" \
    .

echo "  → Image pushed: ${IMAGE_NAME}:latest"

# 4. Deploy to Cloud Run
echo "[4/6] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:latest" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 60 \
    --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO,GCP_PROJECT_ID=${PROJECT_ID}" \
    --project="${PROJECT_ID}" \
    --quiet

# Capture the Cloud Run service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(status.url)")

echo "  → Cloud Run URL: ${SERVICE_URL}"

# 5. Deploy frontend to Firebase Hosting
echo "[5/6] Deploying frontend to Firebase Hosting..."
firebase deploy --only hosting --project="${PROJECT_ID}"

# 6. Final verification
echo "[6/6] Verifying deployment endpoints..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health")
echo "  → Health endpoint HTTP status: ${HEALTH_STATUS}"

echo ""
echo "===================================================================="
echo "  GeoMind deployment complete!"
echo ""
echo "  🌐 Firebase Hosting (Frontend):"
echo "     https://${PROJECT_ID}.web.app"
echo ""
echo "  🚀 Cloud Run API:"
echo "     ${SERVICE_URL}"
echo "     ${SERVICE_URL}/docs"
echo "     ${SERVICE_URL}/health"
echo "     ${SERVICE_URL}/predict"
echo "===================================================================="
