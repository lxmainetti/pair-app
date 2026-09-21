from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.pair_bridge import predict_pairs, MOCK
from ..core.stats import build_matrix, build_cross_matrix, inter_scale_r

router = APIRouter()

class InterScaleRequest(BaseModel):
    items_a: list[str]
    items_b: list[str]

@router.post("/api/inter-scale")
def inter_scale(req: InterScaleRequest):
    items_a = [s.strip() for s in req.items_a if s.strip()]
    items_b = [s.strip() for s in req.items_b if s.strip()]
    if len(items_a) < 1 or len(items_b) < 1:
        raise HTTPException(400, "Each scale needs at least 1 item.")
    all_items = list(dict.fromkeys(items_a + items_b))  # deduped, order-preserving
    try:
        pairs = predict_pairs(all_items)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    r_ab = inter_scale_r(items_a, items_b, pairs)
    cross_pairs = [
        {"item1": a, "item2": b, "r": pairs.get((min(a,b), max(a,b)), 0.0)}
        for a in items_a for b in items_b
    ]
    return {
        "items_a": items_a,
        "items_b": items_b,
        "r_ab": r_ab,
        "cross_matrix": build_cross_matrix(items_a, items_b, pairs),
        "matrix_a": build_matrix(items_a, pairs),
        "matrix_b": build_matrix(items_b, pairs),
        "cross_pairs": sorted(cross_pairs, key=lambda x: -abs(x["r"])),
        "mock": MOCK,
    }
