"""
Runs database — which items were run together, in which analysis.

  runs       one row per successful analysis request
  run_items  its items, in the order entered; scale is 'A'/'B' for inter-scale, NULL otherwise

Mock mode (PAIR_MOCK=1) records nothing: its correlations are fake, so the runs aren't real.
A failed write is logged and never fails the request.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .pair_bridge import MOCK

log = logging.getLogger("uvicorn.error")   # shows up in the service log

DATA_DIR = Path(os.environ.get("PAIR_DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
RUNS_DB  = DATA_DIR / "runs.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT    NOT NULL,          -- UTC ISO 8601
    analysis   TEXT    NOT NULL,          -- 'inter-item' | 'inter-scale' | 'consistency'
    n_items    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS run_items (
    run_id   INTEGER NOT NULL REFERENCES runs(id),
    scale    TEXT,                        -- 'A' | 'B' for inter-scale, NULL otherwise
    position INTEGER NOT NULL,            -- 1-based, within its scale
    item     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS run_items_run  ON run_items(run_id);
CREATE INDEX IF NOT EXISTS run_items_item ON run_items(item);
"""


@contextmanager
def _connect():
    """Connection that commits on success, rolls back on error, and always closes."""
    con = sqlite3.connect(RUNS_DB, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        with con:
            yield con
    finally:
        con.close()


def _init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as con:
        con.executescript(_SCHEMA)


def log_run(analysis: str, scales: dict) -> None:
    """Record one run. scales: {None: items} for a single list, {"A": items_a, "B": items_b} for inter-scale."""
    if MOCK:
        return
    try:
        with _connect() as con:
            n = sum(len(v) for v in scales.values())
            cur = con.execute(
                "INSERT INTO runs (created_at, analysis, n_items) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), analysis, n),
            )
            con.executemany(
                "INSERT INTO run_items (run_id, scale, position, item) VALUES (?, ?, ?, ?)",
                [(cur.lastrowid, scale, i, item)
                 for scale, items in scales.items()
                 for i, item in enumerate(items, start=1)],
            )
    except sqlite3.Error:
        log.exception("Runs database write failed (%s)", analysis)


_init()
