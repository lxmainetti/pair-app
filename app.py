"""PAIR webapp — entry point.

Re-exports the real app from backend/app.py, so both of these work
from the repo root:

    uvicorn app:app --host 0.0.0.0 --port 8080
    uvicorn backend.app:app --host 0.0.0.0 --port 8080

Local dev (mock, no model needed):
    PAIR_MOCK=1 uvicorn app:app --reload --port 8080
"""

from backend.app import app  # noqa: F401
