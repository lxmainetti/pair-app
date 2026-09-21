"""
PAIR Backend
============
Serves the three API endpoints and the static frontend.

Run (mock, local dev):
  PAIR_MOCK=1 uvicorn backend.app:app --reload --port 8080

Run (real, server):
  INFERENCE_URL=http://localhost:8081/predict uvicorn backend.app:app --port 8080
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routers import inter_item, inter_scale, consistency
from .core.pair_bridge import MOCK

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="PAIR")

app.include_router(inter_item.router)
app.include_router(inter_scale.router)
app.include_router(consistency.router)

@app.get("/api/status")
def status():
    return {"mock": MOCK}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((FRONTEND / "index.html").read_text(encoding="utf-8"))
