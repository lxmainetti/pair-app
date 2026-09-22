from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.pair_bridge import predict_pairs, MOCK
from ..core.stats import build_matrix, cronbach_alpha, alpha_if_deleted, alpha_interpretation

router = APIRouter()

class ConsistencyRequest(BaseModel):
    items: list[str]

@router.post("/api/consistency")
def consistency(req: ConsistencyRequest):
    items = [s.strip() for s in req.items if s.strip()]
    if len(items) < 2:
        raise HTTPException(400, "Need at least 2 items.")
    try:
        pairs = predict_pairs(items)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    alpha = cronbach_alpha(items, pairs)
    pair_list = [
        {"item1": a, "item2": b, "r": r}
        for (a, b), r in sorted(pairs.items(), key=lambda x: -abs(x[1]))
    ]
    return {
        "items": items,
        "alpha": alpha,
        "interpretation": alpha_interpretation(alpha),
        "alpha_if_deleted": alpha_if_deleted(items, pairs),
        "matrix": build_matrix(items, pairs),
        "pairs": pair_list,
        "mock": MOCK,
    }
