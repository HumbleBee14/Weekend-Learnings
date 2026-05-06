# 03 — Request Batching

## Files

- `CONCEPTS.md` — why GPUs hate small batches, padding waste, head-of-line blocking
- `server.py` — adds a batcher loop that drains an asyncio queue every 10ms or when 8 requests pile up
- `measure.py` — sweeps concurrency 1, 2, 4, 8, 16 and records throughput + p50/p95/p99

## Quickstart

```bash
uvicorn server:app --workers 1 --port 8000
python measure.py
```

## Expected output

```
concurrency  n     tok/s   p50_ms   p95_ms   p99_ms   avg_batch
1            20    32.1    1450     1620     1700     1.00
2            20    58.4    1480     1700     1790     1.95
4            20    95.2    1620     1900     2050     3.80
8            20    140.7   1880     2300     2500     7.50
16           20    142.1   3500     4100     4400     8.00
```

The pattern:
- Throughput climbs with concurrency until the batch fills consistently (avg_batch ≈ MAX_BATCH_SIZE).
- Past that, throughput plateaus (the batch size cap), but **p99 latency climbs hard** — extra requests queue.
- The "knee" — somewhere around concurrency=8 here — is where you trade more latency for not much more throughput.

## Try

- **Send one tiny prompt and one giant prompt at the same time.** The tiny one waits for the giant one to finish — head-of-line blocking. Real workloads see this constantly.
- **Set MAX_WAIT_MS to 100.** Throughput goes up (bigger batches) but TTFT for low-concurrency requests gets noticeably worse.
- **Set MAX_WAIT_MS to 0.** No batching — every request runs alone. Throughput drops to topic-01 levels.
- **Increase MAX_BATCH_SIZE to 32.** Watch GPU memory. At some point you'll OOM. That's the hard cap.

## What this server still does wrong

1. **Padding waste** — a 50-token prompt batched with a 5000-token prompt pads to 5000.
2. **Head-of-line blocking** — fast users wait for slow users.
3. **No KV cache reuse across requests** — every batch starts fresh.

These all get fixed in Level 4 (paged KV cache + continuous batching). The pain you feel here is the motivation.

## What you measured: G1

The first required graph for Project 1. Plot:
- X-axis: batch size (1 → 16)
- Y-axis-left: throughput (tokens/sec)
- Y-axis-right: p99 latency (ms)

Save the data so you can refer back to it after Level 4 — you'll redo this and the curve will look very different.
