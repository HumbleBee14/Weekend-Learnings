# `mini-platform` — architecture

> One-page reference. The rest of Level 7 implements the boxes here.

## Five-box reference

```
┌────────────────────────────────────────────────────────────────┐
│ 1. GATEWAY                                                     │
│    FastAPI ext-proc shim (mini-platform/gateway/)              │
│    - Auth header check                                         │
│    - Per-tenant token rate-limit (Topic 07)                    │
│    - Cancellation propagation (Topic 15)                       │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│ 2. SCHEDULER / ROUTER                                          │
│    KV-cache-aware router (Topic 06)                            │
│    - Block-hash prefix index (SHA-256, 16-token blocks)        │
│    - Multi-objective scoring: prefix_len * w_p + load * w_l    │
│    - WFQ admission (Topic 07)                                  │
│    - FCFS / priority / SJF policy switch (Topic 09)            │
└────────────┬─────────────────────────────────┬─────────────────┘
             │                                 │
┌────────────▼──────────┐           ┌──────────▼────────────────┐
│ 3a. PREFILL workers   │           │ 3b. DECODE workers        │
│     vLLM, KV publish  │  NIXL     │     vLLM, KV consume      │
│     (Project 2 best)  │ <───────> │     (continuous batching) │
└────────────┬──────────┘           └──────────┬────────────────┘
             │                                 │
             └──────────┬──────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────────┐
│ 4. KV TIER                                                     │
│    LMCache: HBM → DRAM → NVMe → Redis/Mooncake (Topic 12)      │
│    Cross-replica coherence: pull-on-demand via NIXL            │
└────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════╗
║ 5. SIDECARS / OPS PLANE                                        ║
║    Prometheus  (Topic 05)   Grafana       KEDA   (Topic 10)    ║
║    OTel Collector (Topic 05)                                   ║
║    Model Registry (Topic 04) - SQLite + safetensors dir        ║
║    Eval pipeline (Topic 03)  - lm-eval-harness                 ║
╚════════════════════════════════════════════════════════════════╝
```

## Data path (one sentence per hop)

1. Client -> Gateway: TLS terminated, `Tenant-Id` resolved, token quota debited, cancellation channel registered.
2. Gateway -> Router (ext-proc / gRPC): prompt body forwarded, headers preserved.
3. Router: tokenize, block-hash, lookup `PrefixStore`, score replicas, pick winner.
4. Router -> Worker: HTTP POST `/v1/chat/completions`, sticky on chosen replica.
5. Worker: continuous-batched prefill or decode; on KV miss, NIXL-pulls from peer or LMCache backend.
6. Worker -> Router -> Gateway -> Client: SSE stream of tokens; cancellation flows back through the same chain.

## Control path

- Trainer (Level 6) writes checkpoint -> Registry (`staged`).
- Eval pipeline picks `staged` checkpoint -> runs lm-eval-harness -> writes scores.
- Regression gate: scores within X% of previous `serving` -> mark `approved`.
- Operator (or canary controller) flips `approved` -> `serving`. Router's model-table reloads.
- Old version transitions `serving` -> `retired` on N-day TTL.

The registry is **never** in the request path. Router holds model->endpoint mappings in memory and refreshes via watch.

## Failure scenarios

**Worker dies mid-decode.** Gateway's request hangs on TCP. Router's health-check trips within 1-2s, replica drops out of the candidate set. In-flight request returns 503; client retries; KV reuse on the new replica is a function of LMCache backend coverage (cold-cluster: re-prefill; warm: pulled from DRAM/NVMe tier in <1s).

**Eval gate fails on new checkpoint.** Registry status transitions `staged -> eval -> rejected`. The flip to `serving` never fires. Operator reads the eval delta in the Grafana panel built on registry metadata. No data-plane impact.

## Cost model (one paragraph)

Dollars are spent on GPU-seconds. GPU-seconds are split between prefill (compute-bound) and decode (memory-bandwidth-bound). KV reuse converts prefill seconds into a cache lookup; continuous batching converts idle decode seconds into amortised throughput; quantisation buys more concurrent requests per GPU. Dollars are saved when you (a) raise prefix hit rate, (b) raise sustained batch size, (c) match precision to hardware (FP8 on Hopper, NVFP4 on Blackwell). The single number that summarises all of this is `$/Mtok` separated for input and output. Topic 13.
