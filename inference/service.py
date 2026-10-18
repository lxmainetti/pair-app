"""
PAIR Inference Service
======================
Calls PAIR's inference module and exposes a single endpoint:

  POST /predict   {"items": ["item1", "item2", ...]}
  → {"pairs": [{"item1": "...", "item2": "...", "r": 0.42}, ...]}

Run (from this folder):
  uvicorn service:app --port 8081

Model: Qwen/Qwen3-Embedding-8B with the siamese checkpoint from PAIR's models/ folder.

Every item's embedding is stored in the embeddings database (see "Embedding store"
below). Items already in it are not embedded again; only new items go to DeepInfra
(or to PAIR's local embedder).

Env vars:
  PAIR_REPO          Path to PAIR git checkout (server). Not needed if PAIR is pip-installed.
  DEEPINFRA_API_KEY  When set, new items are embedded via DeepInfra (PAIR's
                     backend="deepinfra"). When unset, PAIR embeds them locally.
                     Same model and instruction as training either way.
  PAIR_DATA_DIR      Folder for the SQLite databases (default: <repo>/data).
"""

import logging
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # picks up .env if present; global env vars take precedence

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("uvicorn.error")   # shows up in the service log

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
    _mod.predict = _mod.predict_correlations   # the pip package exports it under this name
    _pair_mod = _mod
    return _pair_mod

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID          = "Qwen/Qwen3-Embedding-8B"
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
DATA_DIR          = Path(os.environ.get("PAIR_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
EMBEDDINGS_DB     = DATA_DIR / "embeddings.sqlite"

# Instruction PAIR embeds with (PSYCHO_INSTRUCT in PAIR's embedding_functions.py).
# Embedding itself happens inside PAIR; this string is only the embedding-store
# key, so it must change whenever PAIR's instruction does.
_QWEN3_TASK = (
    "Given a psychometric scale item, represent its latent psychological "
    "constructs for predicting inter-item response correlations."
)


# ── Embedding store ───────────────────────────────────────────────────────────
# One row per (model, instruction, item). A vector is only reused for the exact
# same model + instruction + item text, so changing either never serves stale
# embeddings — it just embeds again. Vectors are little-endian float32 blobs.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model       TEXT    NOT NULL,
    instruction TEXT    NOT NULL,
    item        TEXT    NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB    NOT NULL,
    source      TEXT    NOT NULL,              -- 'deepinfra' | 'local'
    first_seen  TEXT    NOT NULL,              -- UTC ISO 8601
    last_seen   TEXT    NOT NULL,
    times_seen  INTEGER NOT NULL DEFAULT 1,    -- runs the item appeared in
    PRIMARY KEY (model, instruction, item)
);
"""

@contextmanager
def _connect():
    """Connection that commits on success, rolls back on error, and always closes."""
    con = sqlite3.connect(EMBEDDINGS_DB, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        with con:
            yield con
    finally:
        con.close()

def _init_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as con:
        con.executescript(_SCHEMA)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _load_cached(items: list[str]) -> dict:
    """{item: vector} for the items already stored for this model + instruction."""
    marks = ",".join("?" * len(items))
    with _connect() as con:
        rows = con.execute(
            f"SELECT item, vector FROM embeddings WHERE model = ? AND instruction = ? AND item IN ({marks})",
            (MODEL_ID, _QWEN3_TASK, *items),
        ).fetchall()
    return {item: np.frombuffer(blob, dtype="<f4").copy() for item, blob in rows}

def _save(cached: list[str], new: dict, source: str) -> None:
    """Insert newly embedded items; bump last_seen/times_seen for reused ones."""
    now = _now()
    with _connect() as con:
        con.executemany(
            "INSERT INTO embeddings (model, instruction, item, dim, vector, source, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (model, instruction, item) DO UPDATE SET "
            "last_seen = excluded.last_seen, times_seen = times_seen + 1",
            [(MODEL_ID, _QWEN3_TASK, item, vec.size, vec.astype("<f4").tobytes(), source, now, now)
             for item, vec in new.items()],
        )
        con.executemany(
            "UPDATE embeddings SET last_seen = ?, times_seen = times_seen + 1 "
            "WHERE model = ? AND instruction = ? AND item = ?",
            [(now, MODEL_ID, _QWEN3_TASK, item) for item in cached],
        )

_init_store()

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


# ── Embedding ──────────────────────────────────────────────────────────────────
def _embed(pair, items: list[str]) -> dict:
    """PAIR's own embedder: via DeepInfra if DEEPINFRA_API_KEY is set, else locally.
    Same model and instruction as training either way; return {item: np.ndarray}."""
    backend = "api" if DEEPINFRA_API_KEY else None
    df = pair.embed(items, model=MODEL_ID, backend=backend)
    mat = df.drop("item").to_numpy().astype(np.float32)
    return dict(zip(df["item"].to_list(), mat))


def _embeddings_for(pair, items: list[str]) -> dict:
    """Stored vectors where available; embed and store the rest."""
    try:
        cached = _load_cached(items)
    except sqlite3.Error:
        log.exception("Embedding store read failed; embedding all items")
        cached = {}

    missing = [t for t in items if t not in cached]
    new = {}
    if missing:
        source = "deepinfra" if DEEPINFRA_API_KEY else "local"
        try:
            new = _embed(pair, missing)
        except Exception as e:
            label = "DeepInfra embedding" if DEEPINFRA_API_KEY else "Local embedding"
            raise HTTPException(500, f"{label} error: {e}")
    else:
        source = "cache"

    try:
        _save([t for t in items if t in cached], new, source)
    except sqlite3.Error:
        log.exception("Embedding store write failed")
    log.info("Embeddings: %d from store, %d new (%s)", len(cached), len(new), source)
    return {**cached, **new}


# ── Inference ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    items = list(dict.fromkeys(s.strip() for s in req.items if s.strip()))
    if len(items) < 2:
        raise HTTPException(400, "Need at least 2 items.")
    if len(items) > 50:
        raise HTTPException(400, "Max 50 items per request.")

    try:
        pair = _get_pair()
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    embeddings = _embeddings_for(pair, items)

    try:
        df = pair.predict(items=items, embeddings=embeddings, model=MODEL_ID)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    pairs = [
        PairResult(item1=row[0], item2=row[1], r=round(float(row[2]), 4))
        for row in df.iter_rows()
    ]
    return PredictResponse(pairs=pairs)


@app.get("/health")
def health():
    with _connect() as con:
        stored = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    return {
        "status": "ok",
        "model": MODEL_ID,
        "embeddings": "deepinfra" if DEEPINFRA_API_KEY else "local",
        "stored_items": stored,
    }
