# 15 — Reasoning-Aware Serving

## What changes when reasoning models hit your platform

R1, the o-series, Kimi K2, Claude thinking-mode, GPT-OSS, DeepSeek R1 — these models emit very long, *highly variable* outputs. Kimi K2.6 burns 98K-token reasoning budgets per task. The Artificial Analysis 2026 benchmark spans 160M reasoning tokens. Output-token distributions look nothing like classic chat:

```
Chat (2024)            Reasoning (2026)
────────────────       ──────────────────
mean      ~120 tok      ~6,000 tok
p99       ~600 tok      ~80,000 tok
duration  ~3s           ~minutes
```

Every assumption in classic LLM serving was made for the left column. The right column breaks them. This topic enumerates the breakages and the fixes.

## What breaks

### 1. Output-length variance breaks throughput models

A request that holds a decode slot for minutes is not in the same class as one that finishes in 100ms. Continuous batching handles it (it's just one slow stream among many), but **autoscaler signals based on `num_running` lie**: a long-running tail of slow decodes makes `num_running` look healthy while the *new arrival* queue is exploding.

Fix: scale on `time_in_queue` rather than `num_requests_waiting` count alone. Or: use the LMCache / KV-pressure signal that captures actual saturation (Topic 10).

### 2. Decode-heavy ratio inverts PD disaggregation

Classic chat: prefill ~50% of the work. Reasoning: decode is 95%+ of the work because outputs are 50-100x longer than inputs. Disaggregated inference (Level 5 Topic 8) starts to look very lopsided:

```
prefill workers : decode workers
classic chat:     1 : 1
reasoning:        1 : 8 (or more)
```

Fix: variant autoscaler (Topic 10) that scales prefill and decode pools independently.

### 3. KV cache pressure increases monotonically

Each in-flight reasoning request occupies KV blocks for the full reasoning duration. With 80K-token outputs and 16-token blocks, that's 5,000 blocks per concurrent request. A 32-concurrency engine on H100 may be holding 160K blocks just for active reasoning, before any prefix cache.

Fix: NVMe / LMCache offload becomes mandatory. Per-replica KV pressure (`vllm:gpu_cache_usage_perc`) is a real autoscaling signal — scale up *before* it pegs.

### 4. Naive cost projection is wrong by orders of magnitude

A token-budget projection that assumed 200-token outputs is off by 100x for reasoning workloads. Per-request cost can run from $0.01 to $5 depending on reasoning depth.

Fix: surface reasoning-budget knobs to tenants and bill / budget on output token count, not request count. Forecast cost using p99 output length, not mean.

### 5. Cancellation semantics matter

Users abandon long traces. Without cancellation propagation, the engine keeps decoding for minutes after the client has disconnected. Decode slots get stuck on dead connections.

This is the single most commonly missed feature. It's also the cheapest win — propagate the cancel through gateway -> router -> engine and decode slots free immediately.

## Reasoning budgets — the right user-facing knob

Surface explicit budgets per request. The names converge:

| Knob | Semantics |
|---|---|
| `max_tokens` | Hard cap on total output. |
| `max_thinking_tokens` / `reasoning_max_tokens` | Cap on the reasoning trace specifically. |
| `reasoning_effort` | Coarse mode: `low | medium | high`. Maps to internal budgets per model. |
| `preserve_thinking` | Whether to expose the reasoning trace to the client (Kimi K2). |
| Agent step caps | For agentic flows: max iterations / max tool calls. |

Enforce these at the **gateway** for cost control (Topic 07's quota system extends here). Pass them through to the engine for actual decoding behaviour. Reject requests that exceed tenant tier budgets up front.

## Cancellation propagation — full path

The cancellation must flow all the way to the engine's decode loop. Skipping any link wastes GPU-seconds.

```
client disconnects
  │   asyncio detects EOF on the SSE stream
  │
  ▼
gateway: detect EOF -> cancel coroutine
  │   propagate via the FastAPI request.is_disconnected() check
  │
  ▼
router: upstream connection closed -> close() on aiohttp/httpx stream
  │   the upstream POST to vLLM is aborted
  │
  ▼
engine: TCP RST on the request stream -> scheduler removes request
  │   vLLM V1 already does this if the upstream connection actually closes
  │
  ▼
decode slot freed; KV blocks freed; LMCache demotes the trailing blocks
```

Any layer that caches/buffers without cancellation awareness becomes the failure point. In Python, the typical bug is `BackgroundTask`s that hold a reference to the request and prevent garbage collection.

vLLM V1 propagates cancellation correctly *if* the upstream connection actually closes. Most failures are *upstream of vLLM* — a router that swallows the disconnect, a gateway that holds the response open in a buffer.

References:
- vLLM cancellation handling — https://docs.vllm.ai/en/latest/
- httpx streaming and cancellation — https://www.python-httpx.org/async/

## Architecture deltas for reasoning-heavy workloads

```
Gateway
  - reasoning_effort enforcement per tenant tier
  - max_thinking_tokens per request
  - cancellation: detect_disconnect() check in stream loop
  │
  ▼
Router
  - PD-aware routing: send prefill to prefill pool, then decode-only to decode pool
  - cancellation: close upstream stream on client EOF
  - hedge LESS aggressively (long traces make hedging ruinously expensive)
  │
  ▼
Engine (vLLM with reasoning support)
  - max_tokens / max_thinking_tokens enforced
  - decode pool sized 4-8x larger than prefill pool (variant autoscaler)
  - LMCache NVMe tier mandatory for active KV
  - cancellation: V1 scheduler frees decode slot on stream close
```

## Hedging policy under reasoning loads

Classic hedging: dispatch a duplicate after p95 to reduce p99 (Topic 08). For reasoning, hedging *doubles* a multi-minute decode. Almost always wrong. Either:

- Disable hedging for `reasoning_effort >= medium`.
- Hedge only the first-token TTFT, never the decode tail.
- Cap hedge expenditure: at most 1 hedge per N seconds across the cluster.

## Build steps

1. Add cancellation propagation to your router. The simplest test: run a workload where 30% of clients disconnect mid-generation. Measure decode-slot recovery time.
2. Surface a `reasoning_effort` knob through the gateway. Enforce per-tenant tier caps.
3. Add the `time_in_queue` panel + a `gpu_cache_usage_perc` panel to your Topic 05 dashboard. Drive a long-output workload; observe.
4. Disable hedging for high-reasoning tier. Re-measure p99 — should be unchanged or better (because you stopped wasting GPU on doubled decodes).
5. Document the architecture deltas in `reports/platform.md`.

## Pitfalls

1. **Autoscaler on `num_requests_waiting` only.** Long decodes hide; queue is short but GPUs are saturated. Add `time_in_queue` and `kv_pressure`.
2. **Hedging on long outputs.** Burns 2x GPU on a multi-minute generation. Disable.
3. **No cancellation propagation.** Half your decode slots sit on abandoned requests.
4. **`max_tokens` not enforced at the gateway.** A free-tier user accidentally requests 200K reasoning tokens and racks up $5 cost in one call.
5. **Ignoring decode-pool sizing.** Reasoning workloads want decode-heavy variant pools; one prefill worker per several decode workers.
6. **PD ratios from 2024.** Old 1:1 prefill:decode ratios over-provision prefill for reasoning workloads.

## References

- vLLM continuous batching scheduler — https://docs.vllm.ai/en/latest/
- llm-d Variant Autoscaler — https://llm-d.ai/docs/architecture
- Artificial Analysis reasoning benchmarks — https://artificialanalysis.ai/
- httpx cancellation — https://www.python-httpx.org/async/
