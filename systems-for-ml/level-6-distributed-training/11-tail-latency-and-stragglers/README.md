# 11 — Tail Latency and Stragglers

## Files

- `CONCEPTS.md` — straggler probability math, sources, mitigations, why goodput is the SLO
- `straggler_inject.py` — DDP loop that adds a configurable per-step delay on one rank; reports p50/p95/p99 step time
- `sweep.sh` — sweeps `slow_ms` from 0 to 200; produces the G11 curve

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 straggler_inject.py --slow_ms 0
torchrun --standalone --nproc_per_node=2 straggler_inject.py --slow_ms 50
bash sweep.sh
```

## Expected output

```
world=2  slow_ms=0 on rank0
  p50 step:  4.21 ms
  p95 step:  4.78 ms
  p99 step:  6.12 ms
  mean    :  4.34 ms

world=2  slow_ms=50 on rank0
  p50 step: 54.30 ms
  p95 step: 55.10 ms
  p99 step: 56.40 ms
  mean    : 54.40 ms
```

The 50 ms straggler dominates the 4 ms native step entirely. p99 is no better than p50 — the slow rank is consistently slow, so the variance is small. That's the worst kind of straggler.

## Try

- Make rank 0 *occasionally* slow (e.g., 10% of steps) and rerun. Now p50 stays low and p99 spikes — classic tail-latency profile.
- Run with 4 GPUs and inject on one rank. The cost is the same — the slowest rank determines step time regardless of N.
- Inject thermal-style throttling: `time.sleep(0.1)` for 5 consecutive steps then 50 normal steps. Track windowed step time. This is what real thermal events look like.

## G11 of Project 3

Plot p50, p95, p99 step time as functions of `slow_ms` (the curve from `sweep.sh`). Add a goodput annotation: useful_throughput = baseline_throughput × p50_baseline / p99_observed.

## Where this goes

- Topic 12 — failure injection turns a chronic straggler into a "drop the rank" event
- Topic 13 — async checkpointing keeps stragglers from compounding into lost work
