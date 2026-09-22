"""
PAIR Inference Service
======================
Calls PAIR's inference module and exposes a single endpoint:

  POST /predict   {"items": ["item1", "item2", ...]}
  → {"pairs": [{"item1": "...", "item2": "...", "r": 0.42}, ...]}

Run (server, PAIR git-cloned):
  PAIR_REPO=/opt/PAIR uvicorn inference.service:app --port 8081

Run (local, PAIR pip-installed):
  uvicorn inference.service:app --port 8081

Model: Qwen/Qwen3-Embedding-8B with the siamese checkpoint from PAIR's models/ folder.

Env vars:
  PAIR_REPO          Path to PAIR git checkout (server). Not needed if PAIR is pip-installed.
  DEEPINFRA_API_KEY  When set, embeddings are fetched from DeepInfra with the Qwen3
                     instruction prefix and passed directly to PAIR (PAIR does not re-embed).
                     When unset, PAIR embeds the items locally.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # picks up .env if present; global env vars take precedence

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── PAIR import (pip-installed takes priority; falls back to git-cloned repo) ──
_pair_mod = None

def _get_pair():
    global _pair_mod
    if _pair_mod is not None:
        return _pair_mod
    try:
        import pair as _p
        _pair_mod = _p
        return _pair_mod
    except ImportError:
        pass

    PAIR_REPO = os.environ.get("PAIR_REPO")
    if not PAIR_REPO:
        raise RuntimeError(
            "PAIR is not pip-installed and PAIR_REPO is not set. "
            "Either `pip install -e /path/to/PAIR` or set PAIR_REPO=/path/to/PAIR."
        )
    PAIR_REPO = Path(PAIR_REPO)
    for _dir in (
        PAIR_REPO / "code",
        PAIR_REPO / "code" / "modelling",
        PAIR_REPO / "code" / "data_prep" / "helper_functions",
    ):
        sys.path.insert(0, str(_dir))
    import importlib.util
    _spec = importlib.util.spec_from_file_location("pair_inference", PAIR_REPO / "code" / "inference.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _pair_mod = _mod
    return _pair_mod

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID          = "Qwen/Qwen3-Embedding-8B"
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")

# Qwen3 instruction prefix — matches PAIR's training setup exactly
_QWEN3_TASK   = (
    "Given a psychometric scale item, represent its latent psychological "
    "constructs for predicting response correlations and factor structures."
)
_QWEN3_PREFIX = f"Instruct: {_QWEN3_TASK}\nQuery: "

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


# ── DeepInfra embedding ────────────────────────────────────────────────────────
def _embed_via_deepinfra(items: list[str]) -> dict:
    """Call DeepInfra with the Qwen3 instruction prefix; return {item: np.ndarray}."""
    from openai import OpenAI
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    prefixed = [f"{_QWEN3_PREFIX}{item}" for item in items]
    vectors = []
    for i in range(0, len(prefixed), 100):
        batch = prefixed[i : i + 100]
        resp = client.embeddings.create(model=MODEL_ID, input=batch)
        # sort by index to guarantee order
        vectors.extend([d.embedding for d in sorted(resp.data, key=lambda x: x.index)])
    return {item: np.asarray(vec, dtype=np.float32) for item, vec in zip(items, vectors)}


# ── Inference ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    items = [s.strip() for s in req.items if s.strip()]
    if len(items) < 2:
        raise HTTPException(400, "Need at least 2 items.")
    if len(items) > 50:
        raise HTTPException(400, "Max 50 items per request.")

    try:
        pair = _get_pair()
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    kwargs = dict(model=MODEL_ID)

    if DEEPINFRA_API_KEY:
        # Embed ourselves (with instruction prefix), pass vectors straight to PAIR
        try:
            emb_dict = _embed_via_deepinfra(items)
        except Exception as e:
            raise HTTPException(500, f"DeepInfra embedding error: {e}")
        kwargs["embeddings"] = emb_dict

    try:
        df = pair.predict(items=items, **kwargs)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    pairs = [
        PairResult(item1=row[0], item2=row[1], r=round(float(row[2]), 4))
        for row in df.iter_rows()
    ]
    return PredictResponse(pairs=pairs)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "embeddings": "deepinfra" if DEEPINFRA_API_KEY else "local"}
