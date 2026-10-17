# ─────────────────────────────────────────────────────────────────────────────
# GeoMind — Cloud Run Container Image
# Builds a lightweight Python 3.11 image that serves the FastAPI application.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the full application source
COPY . .

# Cloud Run passes the PORT environment variable (default 8080)
ENV PORT=8080
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Expose the port
EXPOSE 8080

# Launch Uvicorn bound to Cloud Run PORT
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
