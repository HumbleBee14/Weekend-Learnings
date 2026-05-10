# 08 — Disaggregated Inference

## Files

- `CONCEPTS.md` — why prefill and decode have different bottlenecks, KV transfer mechanisms, KV-cache-aware routing, when disagg helps
- `simulate_disagg.py` — single-process simulator comparing colocated vs disaggregated under tunable workload + transfer costs

## Quickstart

```bash
python simulate_disagg.py --n 200 --qps 8 --total-workers 4 \
                          --n-prefill 1 --n-decode 3
```

## Expected output

```
[co-located, 4 workers]
  agg out tok/s   3850
  TTFT  p50/p95/p99    220 / 480 / 690 ms

[disagg, 1 prefill + 3 decode]
  agg out tok/s   4720
  TTFT  p50/p95/p99    180 / 320 / 410 ms
```

The crossover where disagg starts winning depends on QPS, prompt-length distribution, and KV-transfer cost. The simulator lets you sweep those.

## Try

- **Drop QPS to 1.** Disagg overhead now exceeds the gains. Watch its TTFT get *worse* than colocated.
- **Bump `--kv-transfer-us` to 20.** Models a slow network. Disagg loses harder.
- **Set `--n-prefill 2 --n-decode 2`.** Symmetric pools — typical mistake. Asymmetric (more decode) usually wins.
- **Read** the [DistServe retrospective](https://hao-ai-lab.github.io/blogs/distserve-retro/) — what 18 months of production disagg actually taught the field.

## Where this goes

- Topic 09 — Dynamo and llm-d are the orchestration layer above this primitive
- Level 6 — NCCL/RDMA topic covers how the actual KV bytes move (NIXL, GPUDirect)
- Level 7 — `mini-platform`'s router will mimic the KV-cache-aware routing pattern
