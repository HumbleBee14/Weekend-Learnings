# 01 — Inference Server Basics

## Files in this folder

- `CONCEPTS.md` — read first. Explains prefill vs decode, why the GIL serializes requests, why we use 1 worker.
- `server.py` — the 30-line FastAPI server. Read it line by line; every line has a comment explaining why.
- `measure.py` — measures single, sequential, and concurrent request latency. Shows the GIL story in your own data.

## Quickstart

```bash
# 1. Install dependencies
pip install fastapi uvicorn transformers torch httpx

# 2. Terminal A: start the server
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

# 3. Terminal B: hit it once
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain a hash table in one sentence.", "max_tokens":40}'

# 4. Terminal B: run the measurement script
python measure.py
```

## What you should see

After the warmup, the script prints three blocks. The number that matters is **total wall time**:

```
SINGLE                 ~ 1500 ms   (baseline)
SEQUENTIAL (n=10)      ~ 15000 ms  (≈ 10× single — expected)
CONCURRENT (n=10)      ~ 14000 ms  (≈ same as sequential — surprising?)
```

Concurrent ≈ Sequential is the lesson. Sending requests "in parallel" doesn't help when the server can only run one `model.generate()` at a time. This single fact motivates the next 4 topics.

## Things to try

- **Drop `max_tokens` from 50 to 20.** Total time shrinks proportionally — most of the time is decode.
- **Increase to 200.** It scales roughly linearly. There's no fixed per-request overhead worth speaking of (yet).
- **Run with `--workers 4`.** Now concurrent *does* help — but each worker has its own copy of the model. This isn't real batching; it's brute-force parallelism that 4× your GPU memory.
- **Add a `print(perf_counter())` inside `generate()`** to see request timestamps. You'll see them serialize.

## Where this goes next

In topic 02 we add streaming so users see tokens as they're generated (TTFT — time to first token — drops dramatically even though throughput is unchanged). In topic 03 we add real batching, which is what makes concurrent requests *actually* fast.
