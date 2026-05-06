# 05 — Load Testing

## Files

- `CONCEPTS.md` — what real load looks like (Poisson, mixed lengths, slow clients), why mean is a lie, what a CDF tells you
- `locustfile.py` — Locust scenario with short and long prompts, weighted 3:1
- `plot_cdf.py` — reads Locust's CSV output and plots G2

## Quickstart

```bash
pip install locust matplotlib

# Terminal A: topic-03 batched server
cd ../03-request-batching && uvicorn server:app --workers 1 --port 8000

# Terminal B: interactive mode (web UI at http://localhost:8089)
locust -f locustfile.py --host http://localhost:8000

# Or headless (the right way to get clean CSV data for the report)
locust -f locustfile.py --host http://localhost:8000 \
    --headless --users 16 --spawn-rate 4 --run-time 5m --csv g2

# Plot the CDF
python plot_cdf.py g2
```

## What to look for

In the Locust web UI:
- **RPS** (requests per second) — should stabilize after the spawn ramp
- **Median, p95, p99** — the three numbers that matter
- **Failure rate** — at saturation this becomes nonzero

Run multiple sweeps:
- 8 users → 16 users → 32 users → 64 users
- Note where p99 first crosses your target SLO (pick something — say 2000ms)
- That user count is your **saturation point**

## What goes in the report

G2 caption:

> **Setup:** topic-03 batched server, Qwen2.5-0.5B on CPU, 16 concurrent users, 5 minutes, prompt mix 3:1 short:long, max_tokens 40 or 100.
> **Observation:** median 1450ms, p95 2200ms, p99 3100ms, p99.9 8400ms. Failure rate 0%.
> **Insight:** the long-tail divergence above p95 is head-of-line blocking — short requests stuck behind long ones in the same batch. Saturation point is ~24 users for a 2-second p99 SLO.

## Try

- **Push to 100 users.** What's the first thing that breaks? Memory? Timeouts? Connection refused?
- **Add a 3rd prompt class with `max_tokens=500`.** Watch the long tail get worse — those requests dominate batches.
- **Run with the topic-01 server (no batching).** Compare. The CDF shape is dramatically different.
