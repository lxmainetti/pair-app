"""
pair_bridge.py — backend's only contact point with the inference pod.

Real mode : POST to INFERENCE_URL (default http://localhost:8081/predict)
Mock mode : PAIR_MOCK=1  →  deterministic per-pair random correlations,
            same items always produce the same r (keyed by sorted pair hash).
"""

import itertools
import os
import random
from typing import Optional

import httpx

# ── Config ─────────────────────────────────────────────────────────────────────
MOCK          = os.environ.get("PAIR_MOCK", "0") not in ("0", "", "false", "False")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8081/predict")
TIMEOUT       = float(os.environ.get("INFERENCE_TIMEOUT", "120"))   # seconds


# ── Mock helpers ───────────────────────────────────────────────────────────────
def _mock_r(a: str, b: str) -> float:
    key = (min(a, b), max(a, b))
    rng = random.Random(hash(key) & 0xFFFF_FFFF)
    return round(rng.uniform(-0.85, 0.95), 4)


# ── Public API ─────────────────────────────────────────────────────────────────
def predict_pairs(items: list[str]) -> dict[tuple[str, str], float]:
    """
    Return {(item_a, item_b): r} for every unique ordered pair (a < b).
    Symmetric: get_r() in stats.py handles the reverse lookup.
    """
    items = [s.strip() for s in items if s.strip()]

    if MOCK:
        return {
            (a, b): _mock_r(a, b)
            for a, b in itertools.combinations(items, 2)
        }

    # Real: call inference service
    try:
        resp = httpx.post(
            INFERENCE_URL,
            json={"items": items},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach inference service at {INFERENCE_URL}. "
            "Is it running? (uvicorn inference.service:app --port 8081)"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Inference service error {e.response.status_code}: {e.response.text}")

    data = resp.json()
    pairs: dict[tuple[str, str], float] = {}
    for p in data["pairs"]:
        a, b = p["item1"], p["item2"]
        key = (min(a, b), max(a, b))
        pairs[key] = float(p["r"])
    return pairs
