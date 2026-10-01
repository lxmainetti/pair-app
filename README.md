# PAIR web app

Web interface for [PAIR](https://github.com/lxmainetti/PAIR) (Predicting Associations of Inter-item Relationships), which predicts how questionnaire items will correlate from their wording alone. Enter items to get predicted inter-item correlations, inter-scale correlations and Cronbach's α.

## Layout

```
app.py               FastAPI app: the /api routes plus the static frontend
backend/
  core/pair_bridge.py  the only code that talks to the inference service
  core/stats.py        matrices, inter-scale r, Cronbach's α (pure math)
  core/run_log.py      records which items were run (data/runs.sqlite)
  routers/             /api/inter-item, /api/inter-scale, /api/consistency
frontend/            plain HTML/CSS/JS, no build step
inference/service.py model service: embeds items, runs PAIR's siamese network
tools/export_db.py   exports both databases to Parquet
```

The web app and the inference service are separate processes. The web app sends item lists to the service on `INFERENCE_URL`. The service embeds each new item with Qwen3-Embedding-8B, stores the vector in `data/embeddings.sqlite`, and returns a predicted *r* for every pair.

## Running locally

```bash
pip install -r requirements.txt
```

**Mock mode** returns deterministic fake correlations, so you need neither the model nor the inference service:

```bash
PAIR_MOCK=1 uvicorn app:app --reload --port 8080
```

**With the model**, install PAIR first (`pip install -e /path/to/PAIR`, or set `PAIR_REPO`), copy `.env.example` to `.env`, and start both processes:

```bash
uvicorn inference.service:app --port 8081
uvicorn app:app --port 8080
```

## Configuration

| Variable | Used by | Default |
|---|---|---|
| `PAIR_MOCK` | web app | `0` |
| `INFERENCE_URL` | web app | `http://localhost:8081/predict` |
| `INFERENCE_TIMEOUT` | web app | `1800` seconds |
| `DEEPINFRA_API_KEY` | inference | unset: embed locally with PAIR |
| `PAIR_REPO` | inference | unset: use the pip-installed `pair` |
| `PAIR_DATA_DIR` | both, export tool | `./data` |

Keep the inference service bound to localhost. It has no authentication, and with `DEEPINFRA_API_KEY` set every new item costs an embedding call.

## Data

Every real run (not mock mode) stores its items in `data/runs.sqlite`, and every item's embedding is stored in `data/embeddings.sqlite`. The analysis pages tell users this. `data/` is git-ignored. To export both databases to Parquet, run `python tools/export_db.py --out DIR`.
