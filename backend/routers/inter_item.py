from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.pair_bridge import predict_pairs, MOCK
from ..core.stats import build_matrix
from ..core.run_log import log_run

router = APIRouter()

class InterItemRequest(BaseModel):
    items: list[str]

@router.post("/api/inter-item")
def inter_item(req: InterItemRequest):
    items = [s.strip() for s in req.items if s.strip()]
    if len(items) < 2:
        raise HTTPException(400, "Need at least 2 items.")
    try:
        pairs = predict_pairs(items)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    log_run("inter-item", {None: items})
    matrix = build_matrix(items, pairs)
    pair_list = [
        {"item1": a, "item2": b, "r": r}
        for (a, b), r in sorted(pairs.items(), key=lambda x: -abs(x[1]))
    ]
    return {"items": items, "matrix": matrix, "pairs": pair_list, "mock": MOCK}
