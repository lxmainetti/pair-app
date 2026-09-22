"""
Export PAIR's databases to Parquet (read-only; the databases are never modified).

  python tools/export_db.py --out DIR    write the Parquet files into DIR
  python tools/export_db.py | ...        stream them as a tar archive on stdout
                                         (what the `export_db` PowerShell command uses)

Files:
  runs.parquet                   one row per item per run:
                                 run_id, created_at, analysis, n_items, scale, position, item
  embeddings_<model>.parquet     one row per stored item: item, model, instruction, source,
                                 first_seen, last_seen, times_seen, emb1 … embN
                                 (PAIR's format plus metadata — for pair.predict(embeddings=...),
                                  select item + emb columns: df.select("item", pl.col("^emb\\d+$")))

Times are UTC. Each database is read inside one transaction, so the export is a
consistent snapshot even while PAIR is in use.
"""

import argparse
import os
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(os.environ.get("PAIR_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")


def connect(name: str):
    path = DATA_DIR / f"{name}.sqlite"
    if not path.exists():
        return None
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    con.execute("BEGIN")   # one read transaction = one consistent snapshot
    return con


def export_runs(out: Path) -> list[str]:
    con = connect("runs")
    if con is None:
        return []
    rows = con.execute(
        "SELECT r.id, r.created_at, r.analysis, r.n_items, ri.scale, ri.position, ri.item "
        "FROM runs r JOIN run_items ri ON ri.run_id = r.id "
        "ORDER BY r.id, ri.scale, ri.position"
    ).fetchall()
    con.close()
    schema = {"run_id": pl.Int64, "created_at": pl.Utf8, "analysis": pl.Utf8, "n_items": pl.Int64,
              "scale": pl.Utf8, "position": pl.Int64, "item": pl.Utf8}
    df = pl.DataFrame(rows, schema=schema, orient="row")
    df.write_parquet(out / "runs.parquet")
    log(f"runs.parquet: {df['run_id'].n_unique()} runs, {df.height} rows")
    return ["runs.parquet"]


def export_embeddings(out: Path) -> list[str]:
    con = connect("embeddings")
    if con is None:
        return []
    written = []
    models = [m for (m,) in con.execute("SELECT DISTINCT model FROM embeddings ORDER BY model")]
    for model in models:   # one file per model: different models have different dimensions
        rows = con.execute(
            "SELECT item, model, instruction, source, first_seen, last_seen, times_seen, dim, vector "
            "FROM embeddings WHERE model = ? ORDER BY first_seen, item", (model,)
        ).fetchall()
        dims = {r[7] for r in rows}
        if len(dims) != 1:
            raise SystemExit(f"{model}: stored vectors have different dimensions {sorted(dims)}")
        dim = dims.pop()
        mat = np.frombuffer(b"".join(r[8] for r in rows), dtype="<f4").reshape(len(rows), dim)
        meta = pl.DataFrame(
            [r[:7] for r in rows], orient="row",
            schema={"item": pl.Utf8, "model": pl.Utf8, "instruction": pl.Utf8, "source": pl.Utf8,
                    "first_seen": pl.Utf8, "last_seen": pl.Utf8, "times_seen": pl.Int64},
        )
        emb = pl.from_numpy(mat, schema=[f"emb{i + 1}" for i in range(dim)], orient="row")
        name = f"embeddings_{model.replace(':', '-').replace('/', '-')}.parquet"
        pl.concat([meta, emb], how="horizontal").write_parquet(out / name)
        log(f"{name}: {len(rows)} items x {dim} dims")
        written.append(name)
    con.close()
    return written


def log(msg: str) -> None:
    print(msg, file=sys.stderr)   # stderr, so it never mixes into the tar stream


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="write the Parquet files into this folder instead of streaming a tar")
    args = ap.parse_args()

    if args.out is None and sys.stdout.isatty():
        ap.error("stdout is a terminal; pass --out DIR (or pipe the output into a file)")

    with tempfile.TemporaryDirectory() as tmp:
        out = args.out or Path(tmp)
        out.mkdir(parents=True, exist_ok=True)
        files = export_runs(out) + export_embeddings(out)
        if not files:
            raise SystemExit(f"No databases found in {DATA_DIR}")
        if args.out is None:
            with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
                for name in files:
                    tar.add(out / name, arcname=name)
        else:
            log(f"Written to {out.resolve()}")


if __name__ == "__main__":
    main()
