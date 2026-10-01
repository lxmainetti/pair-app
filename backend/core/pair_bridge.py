"""
pair_bridge.py — backend's only contact point with the inference pod.

Real mode : POST to INFERENCE_URL (default http://localhost:8081/predict)
Mock mode : PAIR_MOCK=1  →  deterministic per-pair random correlations,
            same items always produce the same r (keyed by sorted pair hash).
"""

import itertools
import logging
import os
import random
import zlib

import httpx

log = logging.getLogger("uvicorn.error")   # shows up in the service log

# ── Config ─────────────────────────────────────────────────────────────────────
MOCK          = os.environ.get("PAIR_MOCK", "0") not in ("0", "", "false", "False")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8081/predict")
TIMEOUT       = float(os.environ.get("INFERENCE_TIMEOUT", "1800"))   # seconds
MAX_ITEMS     = 50   # the inference service's per-request limit


# ── Mock helpers ───────────────────────────────────────────────────────────────
def _mock_r(a: str, b: str) -> float:
    key = f"{min(a, b)}\x00{max(a, b)}"
    rng = random.Random(zlib.crc32(key.encode()))   # not hash(): str hashes change every process
    return round(rng.uniform(-0.85, 0.95), 4)


# ── Public API ─────────────────────────────────────────────────────────────────
def predict_pairs(items: list[str]) -> dict[tuple[str, str], float]:
    """
    Return {(item_a, item_b): r} for every unique ordered pair (a < b).
    Symmetric: get_r() in stats.py handles the reverse lookup.
    """
    items = [s.strip() for s in items if s.strip()]
    if len(set(items)) > MAX_ITEMS:
        raise ValueError(f"At most {MAX_ITEMS} different items per run.")

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
    # These messages reach the browser, so details (URL, upstream errors) only go to the log.
    except httpx.ConnectError:
        log.error("Cannot reach inference service at %s. Is it running? "
                  "(uvicorn inference.service:app --port 8081)", INFERENCE_URL)
        raise RuntimeError("Inference service unavailable.")
    except httpx.TimeoutException:
        log.error("Inference service at %s timed out after %ss", INFERENCE_URL, TIMEOUT)
        raise RuntimeError("Inference service timed out.")
    except httpx.HTTPStatusError as e:
        log.error("Inference service error %s: %s", e.response.status_code, e.response.text)
        raise RuntimeError(f"Inference service error {e.response.status_code}.")

    data = resp.json()
    pairs: dict[tuple[str, str], float] = {}
    for p in data["pairs"]:
        a, b = p["item1"], p["item2"]
        key = (min(a, b), max(a, b))
        pairs[key] = float(p["r"])
    return pairs
