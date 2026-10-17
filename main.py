"""
GeoMind Application Entrypoint.

Allows running the API using any of:
    uvicorn main:app --reload
    uvicorn api.main:app --reload --port 8080
    python main.py
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Starting GeoMind API on http://{host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
