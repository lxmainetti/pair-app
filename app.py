"""
PAIR webapp
===========
Serves the three API endpoints and the static frontend. Run from the repo root.

Local dev (mock, no model needed):
    PAIR_MOCK=1 uvicorn app:app --reload --port 8080

Server (inference service on :8081):
    uvicorn app:app --host 0.0.0.0 --port 8080
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.routers import inter_item, inter_scale, consistency
from backend.core.pair_bridge import MOCK

FRONTEND = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="PAIR")

app.include_router(inter_item.router)
app.include_router(inter_scale.router)
app.include_router(consistency.router)


@app.get("/api/status")
def status():
    return {"mock": MOCK}


class FrontendFiles(StaticFiles):
    """
    The frontend, with clean URLs (/about serves about.html) and Cache-Control: no-cache,
    so browsers revalidate on every load (a cheap 304 when unchanged). Without that header,
    browsers guess a lifetime and keep serving stale CSS/JS after a deploy.
    """

    def lookup_path(self, path):
        full_path, stat_result = super().lookup_path(path)
        if stat_result is None and path and not Path(path).suffix:
            return super().lookup_path(path + ".html")
        return full_path, stat_result

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


# Static frontend (index.html, about.html, the analysis pages, css/, js/).
# Mounted last so the /api routes above take precedence.
app.mount("/", FrontendFiles(directory=FRONTEND, html=True), name="frontend")
