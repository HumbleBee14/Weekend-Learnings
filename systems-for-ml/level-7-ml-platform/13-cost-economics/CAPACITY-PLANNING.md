# Capacity Planning — Sizing an LLM Service From First Principles

> The question every senior eng gets asked in a planning meeting: *"We're projecting 200 QPS at peak with a p99 target of 2 seconds. How many GPUs do we need?"* This document walks the math.

The `$/Mtok` matrix (CONCEPTS.md) tells you the cost of running a *given* config. Capacity planning answers the reverse: given a target SLO, *what config do you need*. The two views close the loop.

## The five inputs

Before you can size anything, you need numbers — or honest estimates — for:

```
1. Workload shape:
   - Peak QPS (P)                              e.g.  200 req/s
   - Mean input tokens   (I_mean)              e.g.  800 tokens
   - Mean output tokens  (O_mean)              e.g.  300 tokens
   - p99 input tokens    (I_p99)               e.g.  4000 tokens
   - p99 output tokens   (O_p99)               e.g.  1500 tokens

2. Quality target:
   - Model size + quant                        e.g.  Llama-3-70B, FP8
   - p99 TTFT target                           e.g.  500 ms
   - p99 end-to-end target                     e.g.  2000 ms

3. Engine perf on chosen hardware (from your Project 2 bake-off — never guess these):
   - Prefill throughput   (tok/s/GPU)          e.g.  35000  tok/s
   - Decode throughput    (tok/s/GPU)          e.g.   1200  tok/s
   - Concurrent decode slots per GPU           e.g.    64

4. Headroom policy:
   - Target steady-state util                  e.g.  70%
   - Failure budget (N+K redundancy)           e.g.  N+1
```

If you don't have step 3 (real numbers from your own bake-off), every other number in this document is fiction. Run the bake-off first.

## The math

### Step 1 — Token throughput required

```
input_tok_per_sec  = P × I_mean = 200 × 800   = 160,000 tok/s
output_tok_per_sec = P × O_mean = 200 × 300   =  60,000 tok/s
```

These are the rates the whole fleet must sustain on average. Peak (above) drives sizing; you'd validate against a separate average-load calc to confirm you're not massively over-provisioned.

### Step 2 — GPU count from throughput

```
GPUs needed for prefill = input_tok_per_sec / (prefill_throughput × util)
                        = 160,000 / (35,000 × 0.70)  =  6.5  →  7 GPUs

GPUs needed for decode  = output_tok_per_sec / (decode_throughput × util)
                        = 60,000  / (1,200  × 0.70)  = 71.4  →  72 GPUs
```

**This is the lesson** — *decode dominates GPU count*. Input tokens are processed in big batched matmuls during prefill (compute-bound, batchable, cheap per token). Output tokens are produced one-at-a-time per request during decode (memory-bandwidth-bound, harder to batch). On most realistic LLM workloads, decode requires 5–20× more GPUs than prefill on the same hardware.

This is also **why disaggregated prefill/decode exists** (Level 5 Topic 08) — you can run a small prefill pool on expensive GPUs and a large decode pool on cheaper GPUs with high memory bandwidth, and each scales independently.

### Step 3 — Concurrency check

Throughput is necessary but not sufficient. You also need enough *decode slots* (KV-cache concurrency) to hold all in-flight requests.

```
mean request lifetime = O_mean / per_request_decode_speed
                      = 300 / (decode_throughput / concurrent_slots)
                      = 300 / (1200 / 64)
                      = 300 / 18.75
                      = 16 seconds

in-flight requests at peak = P × mean_lifetime
                           = 200 × 16
                           = 3,200 concurrent requests

GPUs needed for concurrency = 3200 / 64 slots = 50 GPUs
```

You need the **max** of (throughput-derived count, concurrency-derived count). Here: `max(72, 50) = 72`. On long-context workloads (when p99 input/output is far above mean), the concurrency-derived number is often the binding one.

### Step 4 — p99 headroom

The numbers above are **mean** — they tell you fleet capacity at steady state. For p99 SLOs, you need headroom because:
- p99 requests have ~5× the tokens of mean
- Queue tail (Little's Law) means utilization > 70% blows up p99 latency
- Failures and stragglers add 10-20% tail variance

Two rules of thumb that have survived contact with production:

```
GPUs_p99_safe   = GPUs_throughput × 1.3      ← p99 headroom factor
GPUs_with_HA    = GPUs_p99_safe + K          ← N+K for failures (K usually 1–2)
GPUs_with_warm  = GPUs_with_HA   × 1.2       ← warm pool for autoscale lag
```

Going through:
```
72 GPUs   (throughput at 70% util)
× 1.3     (p99 headroom)        = 94
+ 2       (N+2 redundancy)      = 96
× 1.2     (warm pool)           = 116 GPUs at peak
```

### Step 5 — $/Mtok sanity check

Cross to the cost matrix in CONCEPTS.md. At Llama-3-70B FP8 on H100, you got `$2.47 / Mtok blended`. Tokens served per day at peak:

```
tokens/day = (input + output) × QPS × 86400
           = (800 + 300) × 200 × 86400
           ≈ 19 billion tokens/day
           = 19,000 Mtok/day

cost/day = 19,000 × $2.47 = $46,930/day  ≈  $1.4M/month
```

Now compare to:
- Quantizing to NVFP4 on B200: `$1.41 / Mtok` → ~$0.8M/month
- Adding 30% prefix-cache hit rate (RAG-shaped workload): another ~25% reduction
- Speculative decoding (2× decode throughput): cuts decode GPUs roughly in half

This is **where the bake-off and cost matrix feed back into capacity planning** — you don't just size once, you re-size each time you change a major lever.

## Worked example: full table

Llama-3-70B, 200 QPS peak, p99 TTFT 500ms, p99 e2e 2s, 800/300 token mean.

| Config | Prefill GPUs | Decode GPUs | Concurrency GPUs | Bind | × p99 | + HA | + warm | Final | $/month |
|---|---|---|---|---|---|---|---|---|---|
| FP8 H100 baseline | 7 | 72 | 50 | 72 | 94 | 96 | 116 | **116** | $1.4M |
| FP8 H100 + 30% prefix cache | 5 | 50 | 35 | 50 | 65 | 67 | 81 | **81** | $0.97M |
| FP8 H100 + spec decode (2×) | 7 | 36 | 25 | 36 | 47 | 49 | 59 | **59** | $0.71M |
| NVFP4 B200 baseline | 4 | 36 | 25 | 36 | 47 | 49 | 59 | **59** | $0.85M |
| NVFP4 B200 + everything above | 3 | 18 | 13 | 18 | 24 | 26 | 32 | **32** | $0.45M |
| llama.cpp on CPU EPYC | ∞ | 5000+ | ∞ | ∞ | — | — | — | not viable at this QPS | — |

The last row is the punchline of [Level 5 Topic 05](../../level-5-production-engines/05-llama-cpp-deep-dive/) — CPU inference is competitive on **low-QPS, high-context** workloads, not on 200 QPS chat. Capacity planning is what tells you that.

## Try

- **Plug in your own Project 2 numbers.** Replace the prefill / decode / concurrency cells with measurements from your bake-off. The math doesn't change; only the inputs do.
- **Sweep p99 target from 1s → 5s.** Watch how aggressive the headroom multiplier needs to be to hit a tight p99 — and how cheap loose p99 SLOs become.
- **Stress the input distribution.** Run the same math with `I_p99 = 32000` (long-context RAG). The decode count usually doesn't change much; the concurrency count blows up. This is when you start thinking about KV tiering (Topic 12) and offline batch (Level 5 Topic 11).
- **Failure scenario.** What happens to p99 the moment 1 of 116 GPUs dies? Math: utilization jumps from 70% to 70.6% → barely matters. Lose 10 GPUs? 70% → 76.6% → p99 starts climbing. Lose 30 GPUs? You're SLO-breaching. This is what *Goodput* (Level 6) and N+K sizing exist to prevent.

## Where this goes

- `reports/platform.md` — the "Capacity" section is exactly this calculation, with your measured numbers.
- Level 7 Topic 10 (`autoscaling-keda`) — the autoscaler implements the dynamic version of this math (queue depth → fleet size). Capacity planning is the static version that tells you the autoscaler's min and max bounds.
- Level 7 Topic 11 (`cold-start-and-warmup`) — the warm-pool multiplier (× 1.2 above) is the cost of cold-start mitigation. If you can cold-start in <30s, you can shrink the warm pool.

## The thing nobody tells you

Capacity planning is **wrong on the first pass, every time.** Real workloads have skew (one tenant uses 80% of capacity), bursts (Black Friday for a chatbot), and surprises (a customer starts sending 100K-token prompts). The job isn't to nail the number in one pass — it's to:

1. Get within 2× on the first sizing.
2. Wire observability (Topic 05) that surfaces the *actual* prefill/decode/concurrency utilization.
3. Build autoscaling (Topic 10) that absorbs short-term error.
4. Re-size monthly as the workload evolves.

The senior-eng skill is knowing that the math above is a **floor for the conversation**, not a prediction. Bring numbers, defend assumptions, plan for re-sizing.

## References

- [vLLM benchmark guide](https://docs.vllm.ai/en/latest/performance/benchmarks.html) — how to measure prefill/decode throughput honestly
- Little's Law derivation in [Level 7 Topic 08 (`backpressure-and-queueing`)](../08-backpressure-and-queueing/CONCEPTS.md) — the formal version of the concurrency math above
- [Reddi Vol 2 — *Ops at Scale*](https://mlsysbook.ai/) for the canonical framing
