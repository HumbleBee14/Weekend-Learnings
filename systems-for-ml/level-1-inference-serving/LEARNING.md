# Level 1 — Learning Path

How to actually go through Level 1. Read `README.md` first for the level-wide context. Then follow the topics in order.

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-inference-server-basics/` | 1–2h | A 30-line FastAPI server. The GIL serialization story confirmed in your own data. |
| `02-streaming-tokens/` | 1–2h | SSE streaming. TTFT vs ITL as separate metrics. |
| `03-request-batching/` | 2–3h | Async batcher loop. Padding waste and head-of-line blocking witnessed firsthand. |
| `04-latency-vs-throughput/` | 1h | The throughput-vs-p99 curve plotted (G1 of Project 1). |
| `05-load-testing/` | 1–2h | Locust scenario + latency CDF (G2 of Project 1). Saturation point identified. |
| `06-local-first-touch/` | 1–2h | Ollama side-by-side. A feel for what local serving looks like. |
| `_capstone-mini-serve/` | 2–3h | All of the above stitched into one well-structured server with tests and config. |

## How to use each topic folder

Every topic folder has the same shape:

- **`CONCEPTS.md`** — read first. Direct notes on what's being taught and why.
- **`server.py`** (or similar) — working code with comments explaining each piece.
- **`measure.py`** (or `sweep_and_plot.py` / etc.) — the experiment that makes the concept land.
- **`README.md`** — the quickstart commands and what to look for in the output.

Don't skip the experiment scripts. The point of building a wrong-but-working server is to *measure* the wrongness yourself. The throughput-vs-latency story doesn't land until you see your own data showing the knee.

## What to put in `reports/week1.md`

Project 1 requires:

- **G1** — throughput vs p99 latency curve (from topic 04)
- **G2** — latency CDF at fixed concurrency (from topic 05)
- **Setup → Observation → Insight** captions on both

Your `reports/week1.md` lives at the top of the level folder (you'll create it yourself when you have the data). Format from the outer `systems-for-ml/README.md`.

## After this level

You have a working server. You also have a list of things it does badly:

- One slow request blocks fast ones (head-of-line blocking)
- Padding waste makes mixed-length batches inefficient
- No prefix sharing between requests
- KV cache memory grows linearly with context

These pain points are the motivation for everything in Levels 2, 3, and 4. The fix list is closed at the end of Level 4 with `mini-vllm`.
