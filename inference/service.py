"""
PAIR Inference Service
======================
Calls PAIR's inference module and exposes a single endpoint:

  POST /predict   {"items": ["item1", "item2", ...]}
  → {"pairs": [{"item1": "...", "item2": "...", "r": 0.42}, ...]}

Run:
  PAIR_REPO=/opt/PAIR uvicorn inference.service:app --port 8081
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY (and others) from .env if present

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── PAIR repo paths ────────────────────────────────────────────────────────────
PAIR_REPO   = Path(os.environ.get("PAIR_REPO",   "/opt/PAIR"))
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", str(PAIR_REPO / "models")))
MODEL_ID    = os.environ.get("MODEL_ID", "text-embedding-3-large")
MODEL_DIR   = MODELS_ROOT / MODEL_ID

for _p in (
    PAIR_REPO / "code",
    PAIR_REPO / "code" / "modelling",
    PAIR_REPO / "code" / "data_prep" / "helper_functions",
):
    sys.path.insert(0, str(_p))

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="PAIR Inference Service")


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    items: list[str]

class PairResult(BaseModel):
    item1: str
    item2: str
    r: float

class PredictResponse(BaseModel):
    pairs: list[PairResult]


# ── Inference ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    items = [s.strip() for s in req.items if s.strip()]
    if len(items) < 2:
        raise HTTPException(400, "Need at least 2 items.")
    if len(items) > 50:
        raise HTTPException(400, "Max 50 items per request.")

    if not MODEL_DIR.exists() or not (MODEL_DIR / "dnn_siamese_cor.pt").exists():
        raise HTTPException(500, f"Model checkpoint not found: {MODEL_DIR}")

    try:
        import importlib.util, sys as _sys
        _spec = importlib.util.spec_from_file_location("pair_inference", PAIR_REPO / "code" / "inference.py")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        predict_correlations = _mod.predict_correlations
        to_matrix = _mod.to_matrix
    except ImportError as e:
        raise HTTPException(500, f"Cannot import PAIR inference module: {e}")

    try:
        df = predict_correlations(
            items=items,
            model=MODEL_ID,
            models_root=MODELS_ROOT,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    pairs = [
        PairResult(item1=row[0], item2=row[1], r=round(float(row[2]), 4))
        for row in df.iter_rows()
    ]
    return PredictResponse(pairs=pairs)


@app.get("/health")
def health():
    meta_path = MODEL_DIR / "embedding_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {"status": "ok", "model": MODEL_ID, "meta": meta}
