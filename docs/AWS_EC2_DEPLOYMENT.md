# GeoMind — AWS EC2 Deployment Guide (Ubuntu Linux)

This guide provides step-by-step instructions for deploying the **GeoMind AI Urban Traffic Forecasting** application on an **AWS EC2 Ubuntu** instance, starting from a local Windows development machine.

---

## Architecture Overview

```
                                +-----------------------------------+
                                |            AWS EC2                |
+---------------------+         |  Ubuntu 22.04 / 24.04 LTS         |
| Client Browser      |         |                                   |
| (Local / Public IP) | =======>|  Uvicorn (:8000)                  |
|                     |  HTTP   |    └─ FastAPI (api.main:app)      |
+---------------------+         |         ├─ GET  / (Frontend)      |
                                |         ├─ GET  /health           |
                                |         ├─ GET  /docs             |
                                |         └─ POST /predict          |
                                |              └─ ML Models (.joblib)
                                +-----------------------------------+
```

---

## 1. AWS Account Prerequisites

1. An active AWS account with administrative or EC2 provisioning privileges.
2. Sign in to the [AWS Management Console](https://aws.amazon.com/console/).
3. Choose your preferred AWS Region (e.g., `us-east-1` (N. Virginia), `ap-south-1` (Mumbai), etc.).

---

## 2. Launch an AWS EC2 Instance

1. Navigate to the **EC2 Dashboard** and click **Launch instance**.
2. **Name**: `GeoMind-Production-Server`.
3. **Application and OS Images (Amazon Machine Image)**:
   - Select **Ubuntu**.
   - Choose **Ubuntu Server 22.04 LTS (HVM)** or **Ubuntu Server 24.04 LTS (HVM)**, 64-bit (x86).
4. **Instance Type**:
   - Select `t2.micro` (or `t3.micro`), which is **Free Tier eligible** (1 vCPU, 1 GiB Memory).
   - *Note*: If you run multiple models concurrently, `t3.small` (2 GiB RAM) provides extra headroom, but `t2.micro` works comfortably for our lightweight inference.
5. **Storage**:
   - 15 GiB - 20 GiB gp3 General Purpose SSD (default 8 GiB is often tight after installing packages).

---

## 3. Key Pair Creation

1. In the **Key pair (login)** section, click **Create new key pair**.
2. **Key pair name**: `geomind-key`.
3. **Key pair type**: `RSA`.
4. **Private key file format**: `.pem` (for OpenSSH in PowerShell).
5. Click **Create key pair**.
6. Save the downloaded `geomind-key.pem` file in a known folder on your Windows machine, e.g.:
   `C:\Users\<YourUser>\.ssh\geomind-key.pem`

---

## 4. Security Group Configuration

In the **Network settings** section, configure the firewall rules:

1. Click **Create security group**.
2. Enable the following inbound rules:

| Type | Protocol | Port Range | Source | Purpose |
|------|----------|------------|--------|---------|
| **SSH** | TCP | `22` | `My IP` (or `0.0.0.0/0`) | Secure shell access |
| **Custom TCP** | TCP | `8000` | `0.0.0.0/0` (Anywhere IPv4) | FastAPI & Web Frontend |
| **HTTP** *(Optional)* | TCP | `80` | `0.0.0.0/0` | If configuring Nginx |

3. Click **Launch instance**.
4. Once launched, click on the instance ID and copy the **Public IPv4 address** (e.g., `54.210.12.34`).

---

## 5. Connecting from Windows PowerShell via SSH

1. Open **Windows PowerShell** as Administrator or standard user.
2. (Important Windows permission step): Ensure your private key has restricted permissions:
   ```powershell
   # Navigate to your key folder
   cd C:\Users\<YourUser>\.ssh

   # Set permissions (read-only for current user)
   icacls geomind-key.pem /inheritance:r
   icacls geomind-key.pem /grant:r "$($env:USERNAME):(R)"
   ```
3. Connect to the EC2 instance using SSH:
   ```powershell
   ssh -i "geomind-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
   ```
   *(Type `yes` when prompted to verify host authenticity).*

---

## 6. Updating Ubuntu & Installing Dependencies

Once connected to the remote Ubuntu shell, update packages and install Python:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git curl -y
```

Verify installed versions:
```bash
python3 --version
git --version
```

---

## 7. Cloning the GitHub Repository

Clone your repository from GitHub into the `ubuntu` home directory:

```bash
git clone https://github.com/<your-username>/GeoMind.git
cd GeoMind
```

Verify repository content:
```bash
ls -la
```

---

## 8. Creating & Activating Virtual Environment

Create an isolated Python virtual environment inside the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify that the active Python points to the virtual environment:
```bash
which python
# Output should be: /home/ubuntu/GeoMind/.venv/bin/python
```

---

## 9. Installing Project Dependencies

Upgrade `pip` and install all required runtime dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Running Test Verification

Before running the server, execute the verification tests to ensure that all models (`xgboost`, `random_forest`, `ridge_regression`) and the preprocessor load and infer properly on Linux:

```bash
python tests/run_tests.py
```

Expected output:
```text
Testing model loading...
Status: {'preprocessor': True, 'xgboost': True, 'random_forest': True, 'ridge_regression': True}
...
ALL VERIFICATIONS PASSED SUCCESSFULLY!
```

---

## 11. Starting FastAPI & Uvicorn Manually

Start Uvicorn bound to `0.0.0.0` (all network interfaces) so it can receive public internet traffic:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

You should see:
```text
INFO:     [2026-10-02 12:00:00] INFO - GeoMind - GeoMind API starting (env=production) — loading models...
INFO:     [2026-10-02 12:00:00] INFO - GeoMind -   OK: preprocessor
INFO:     [2026-10-02 12:00:00] INFO - GeoMind -   OK: xgboost
INFO:     [2026-10-02 12:00:00] INFO - GeoMind -   OK: random_forest
INFO:     [2026-10-02 12:00:00] INFO - GeoMind -   OK: ridge_regression
INFO:     [2026-10-02 12:00:00] INFO - GeoMind - GeoMind API ready for traffic forecasting inference.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 12. Testing from Your Web Browser

Open your browser on Windows and navigate to:

1. **Web Dashboard**:
   ```
   http://<YOUR_EC2_PUBLIC_IP>:8000/
   ```
   *The GeoMind interface should load immediately with status "GeoMind API Connected".*

2. **Swagger / Interactive OpenAPI Documentation**:
   ```
   http://<YOUR_EC2_PUBLIC_IP>:8000/docs
   ```

3. **Production Health Check**:
   ```
   http://<YOUR_EC2_PUBLIC_IP>:8000/health
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
     "api_version": "3.1.0"
   }
   ```

4. **Inference Test via cURL (or PowerShell)**:
   ```bash
   curl -X POST "http://<YOUR_EC2_PUBLIC_IP>:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "observation": {
         "date_time": "2026-10-01T08:00:00",
         "temp": 288.15,
         "rain_1h": 0.0,
         "snow_1h": 0.0,
         "clouds_all": 20.0,
         "weather_main": "Clear",
         "road_condition": "Dry / Normal",
         "holiday": "None"
       },
       "model_type": "xgboost"
     }'
   ```

---

## 13. Configuring Background Systemd Service

To keep the application running continuously after you close PowerShell and ensure it automatically restarts on system reboot or unexpected crashes, install the `systemd` service:

1. Copy the service unit file to systemd directory:
   ```bash
   sudo cp deployment/geomind.service /etc/systemd/system/geomind.service
   ```

2. Reload systemd daemon:
   ```bash
   sudo systemctl daemon-reload
   ```

3. Enable the service to start automatically upon EC2 boot:
   ```bash
   sudo systemctl enable geomind.service
   ```

4. Start the service:
   ```bash
   sudo systemctl start geomind.service
   ```

---

## 14. Managing & Monitoring the Service

### Check Service Status
```bash
sudo systemctl status geomind.service
```

### View Live Real-Time Logs
```bash
journalctl -u geomind.service -f
```

### View the Last 100 Log Lines
```bash
journalctl -u geomind.service -n 100 --no-pager
```

### View Application File Logs
```bash
cat logs/$(date +%Y_%m_%d).log
```

### Restart Service
```bash
sudo systemctl restart geomind.service
```

### Stop Service
```bash
sudo systemctl stop geomind.service
```

---

## 15. Updating the Application After a GitHub Push

Whenever you make improvements locally on Windows, commit and push to GitHub, follow this quick 4-step update procedure on EC2:

```bash
# 1. SSH into your EC2 instance
ssh -i "geomind-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>

# 2. Enter directory and pull latest code
cd /home/ubuntu/GeoMind
git pull origin main

# 3. Update dependencies (if requirements changed)
source .venv/bin/activate
pip install -r requirements.txt

# 4. Restart the systemd service
sudo systemctl restart geomind.service

# 5. Verify status
sudo systemctl status geomind.service
```

---

## 16. Optional Production Nginx Reverse Proxy Setup

If you wish to serve the application on standard HTTP Port 80 instead of `:8000`:

1. Install Nginx:
   ```bash
   sudo apt install nginx -y
   ```

2. Create an Nginx site configuration:
   ```bash
   sudo nano /etc/nginx/sites-available/geomind
   ```

3. Paste the following configuration:
   ```nginx
   server {
       listen 80;
       server_name _;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

4. Enable the configuration and restart Nginx:
   ```bash
   sudo ln -sf /etc/nginx/sites-available/geomind /etc/nginx/sites-enabled/
   sudo rm -f /etc/nginx/sites-enabled/default
   sudo nginx -t
   sudo systemctl restart nginx
   ```
   Now you can access `http://<YOUR_EC2_PUBLIC_IP>/` directly on port 80!

---

## 17. Troubleshooting Common Issues

### Issue 1: Port 8000 Timed Out in Browser
- **Cause**: EC2 Security Group is missing an inbound rule for port 8000.
- **Fix**: Go to **EC2 Console** → **Instances** → Select your instance → **Security** tab → Click Security Group → **Edit inbound rules** → Add rule: Type `Custom TCP`, Port `8000`, Source `0.0.0.0/0`.

### Issue 2: `models/ml/*.joblib` Not Found
- **Cause**: Git was ignoring model files or files were not pulled.
- **Fix**: Verify `.gitignore` does not ignore `models/ml/` and run `ls -lh models/ml/` to confirm all 3 joblib files exist.

### Issue 3: `Permission denied (publickey)` on SSH
- **Cause**: Wrong key file or incorrect permissions on Windows.
- **Fix**: Ensure username is `ubuntu` (not `root` or `admin`). Run the `icacls` command shown in Section 5.

### Issue 4: Out of Memory on `t2.micro`
- **Cause**: Compiling packages during pip install.
- **Fix**: Add a 1GB swap space on EC2:
  ```bash
  sudo fallocate -l 1G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

---

## 18. AWS Cost Control & Cleanup

- **Stopping the Instance**: When not demonstrating the project, stop your instance to avoid charges:
  - EC2 Console → Select instance → **Instance state** → **Stop instance**.
  - *IP address will change upon restart unless an Elastic IP is used.*
- **Terminating the Instance**: When your semester demonstration is finished:
  - EC2 Console → Select instance → **Instance state** → **Terminate instance**.
- **Free Tier Warning**: AWS Free Tier offers 750 hours/month of `t2.micro` (or `t3.micro` depending on region) for the first 12 months. Monitor your usage at [AWS Billing Dashboard](https://console.aws.amazon.com/billing/).
