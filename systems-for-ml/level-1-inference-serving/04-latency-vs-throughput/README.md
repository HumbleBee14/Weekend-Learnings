# 04 — Latency vs Throughput

## Files

- `CONCEPTS.md` — the curve, the knee, why latency explodes past it
- `sweep_and_plot.py` — runs the concurrency sweep against the topic-03 server, writes `g1_data.csv` and `g1_plot.png`

## Quickstart

```bash
# Terminal A: run the topic-03 batched server
cd ../03-request-batching && uvicorn server:app --workers 1 --port 8000

# Terminal B
pip install matplotlib  # if you don't have it
python sweep_and_plot.py
```

Output: a CSV of (concurrency, throughput, p50, p95, p99) at each level, plus a PNG.

## Reading the plot

```
throughput  ────────────╮
                  ╱     │ flattens around the knee
            ╱           │
        ╱               │
   ╱                    │
  └────────────────────►
                        ↑
                 p99 latency
                  ╮      ╭── explodes past the knee
                   ╲    ╱
                    ╲  ╱
                ─────╲╱
                  knee
```

The knee is the operating point. Past it, you're paying tail latency for marginal throughput — usually a bad trade.

## What goes in the report

`reports/week1.md` should have G1 with this caption format:

> **Setup:** Qwen2.5-0.5B on M2 Mac CPU, 5-prompt corpus, max_tokens=50, MAX_BATCH_SIZE=8, MAX_WAIT_MS=10.
> **Observation:** throughput rises from 32 tok/s (c=1) to 142 tok/s (c=8), then flattens. p99 climbs from 1700ms to 2500ms over the same range, then rises sharply past c=8.
> **Insight:** the knee is at c=8 — same as MAX_BATCH_SIZE, expected. To push throughput higher we'd need a larger batch cap (memory permitting) or a bigger machine. To push p99 lower at high concurrency we'd need continuous batching to eliminate head-of-line blocking.

This caption format — **Setup → Observation → Insight** — is what every graph in this curriculum follows.
