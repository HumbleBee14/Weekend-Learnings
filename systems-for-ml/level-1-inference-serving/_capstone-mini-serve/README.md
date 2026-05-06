# mini-serve — Level 1 Capstone

Combines everything from topics 01–05 into one well-structured server. This is the artifact you keep extending through Levels 3 and 4 (it becomes `mini-vllm` once you add the paged KV cache).

## What this gives you over the per-topic code

The per-topic servers (01–05) are intentionally minimal — one file each, one lesson each. This capstone shows what production conventions look like when you put it together:

| Convention | Where |
|---|---|
| Twelve-factor config (env vars, no hardcoded values) | `mini_serve/config.py` |
| Typed request/response schemas | `mini_serve/schemas.py` |
| Isolated model loader (testable, swappable) | `mini_serve/model_loader.py` |
| Async batcher with backpressure | `mini_serve/batcher.py` |
| Lifespan-managed startup/shutdown | `mini_serve/app.py` |
| Both blocking and streaming endpoints unified | `app.py` routes |
| Unit-testable bits (schemas) | `tests/test_schemas.py` |
| Smoke test script | `scripts/smoke_test.py` |
| Project metadata | `pyproject.toml` |

## Layout

```
_capstone-mini-serve/
├── mini_serve/
│   ├── __init__.py
│   ├── config.py          # Pydantic Settings — env-driven config
│   ├── schemas.py         # Request/response models
│   ├── model_loader.py    # Load model + tokenizer + pick device
│   ├── batcher.py         # Async micro-batcher (queue + loop)
│   └── app.py             # FastAPI app + routes
├── tests/
│   └── test_schemas.py    # Fast tests that don't need the model
├── scripts/
│   └── smoke_test.py      # End-to-end check
├── pyproject.toml
└── README.md
```

## Run

```bash
cd _capstone-mini-serve
pip install -e ".[dev]"

# Start (env vars override defaults)
MINI_SERVE_MAX_BATCH_SIZE=8 uvicorn mini_serve.app:app --host 0.0.0.0 --port 8000

# In another terminal
python scripts/smoke_test.py

# Tests
pytest -v
```

## Try

```bash
# Override config via env
MINI_SERVE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" \
MINI_SERVE_MAX_BATCH_SIZE=16 \
MINI_SERVE_MAX_WAIT_MS=20 \
uvicorn mini_serve.app:app

# Test backpressure: flood the server, expect some 503s
ab -n 200 -c 200 -p body.json -T 'application/json' http://localhost:8000/generate
```

## What's deliberately *not* here

Things you'd add in a real production service that aren't necessary for the curriculum lesson:

- Auth (API keys, JWT, OAuth)
- Rate limiting at the app layer (do it at the gateway in Level 7)
- Detailed Prometheus metrics (Level 7)
- Distributed tracing (OpenTelemetry — Level 7)
- Graceful in-flight request draining on shutdown
- Multi-worker deployment via Kubernetes (Level 7 + Level 6)
- Paged KV cache and continuous batching (Level 4)

Those all belong in their own levels. The capstone here is the *baseline shape* of a real server — clean enough to extend, simple enough to read in 15 minutes.

## Where this goes

In Level 4 you'll fork this into `mini-vllm` by replacing the naive batcher with a paged KV cache and continuous batching. In Level 7 the `mini-platform` capstone wraps this server in a router + autoscaler + observability stack.
