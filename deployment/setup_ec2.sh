#!/usr/bin/env bash
# ==============================================================================
# GeoMind — AWS EC2 Ubuntu Automated Setup Script
# ==============================================================================
# Run this script on a fresh Ubuntu 22.04 / 24.04 EC2 instance to configure the
# Python virtual environment, install requirements, and verify the ML pipeline.
#
# Usage:
#   chmod +x deployment/setup_ec2.sh
#   ./deployment/setup_ec2.sh
# ==============================================================================

set -euo pipefail

echo "===================================================================="
echo "  GeoMind EC2 Environment Setup"
echo "===================================================================="

# 1. Update package index and install required system packages
echo "[1/5] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

# 2. Setup Python virtual environment
echo "[2/5] Creating Python virtual environment in .venv..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  -> .venv created."
else
    echo "  -> .venv already exists, reusing."
fi

# 3. Upgrade pip and install requirements
echo "[3/5] Installing Python dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# 4. Run test suite to verify ML models and preprocessor
echo "[4/5] Running verification tests..."
.venv/bin/python tests/run_tests.py

# 5. Setup systemd service (optional prompt or instructions)
echo "[5/5] Checking systemd service configuration..."
SERVICE_FILE="deployment/geomind.service"
if [ -f "$SERVICE_FILE" ]; then
    echo "  Installing systemd service to /etc/systemd/system/geomind.service..."
    sudo cp "$SERVICE_FILE" /etc/systemd/system/geomind.service
    sudo systemctl daemon-reload
    sudo systemctl enable geomind.service
    sudo systemctl restart geomind.service
    echo "  -> Service status:"
    sudo systemctl status geomind.service --no-pager --lines=5
fi

echo "===================================================================="
echo "  GeoMind setup complete!"
echo "  Access endpoints:"
echo "    http://<YOUR_EC2_PUBLIC_IP>:8000/"
echo "    http://<YOUR_EC2_PUBLIC_IP>:8000/docs"
echo "    http://<YOUR_EC2_PUBLIC_IP>:8000/health"
echo "===================================================================="
