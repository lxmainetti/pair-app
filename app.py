"""PAIR webapp — entry point.

Local dev (mock, no model needed):
    PAIR_MOCK=1 uvicorn app:app --reload --port 8080

Production:
    uvicorn app:app --host 0.0.0.0 --port 8080
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.pair_bridge import list_models, MOCK
from routers import inter_item, inter_scale, consistency

app = FastAPI(title="PAIR", description="Pairwise item-correlation predictions")

app.include_router(inter_item.router)
app.include_router(inter_scale.router)
app.include_router(consistency.router)


@app.get("/api/models")
def get_models():
    return {"models": list_models(), "mock": MOCK}


STATIC = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))
