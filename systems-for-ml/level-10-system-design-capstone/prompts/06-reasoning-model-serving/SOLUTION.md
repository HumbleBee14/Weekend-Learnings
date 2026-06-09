# Prompt 06 — Worked Solution

> Open this only after attempting the prompt cold. This is one defensible design, not the only one.

## 1. Clarifying questions (the first 3 minutes)

A senior candidate asks these *before* drawing anything. They scope the design and signal you've actually shipped a reasoning model into production:

1. **Output distribution shape.** "Mean 2K, p99 30K" — is the tail bimodal (most short, occasional thinker) or smooth long-tail? (Bimodal lets you do early-classification routing; smooth tail makes you size for the worst case continuously.)
2. **Is the reasoning trace returned to the user, hidden, or summarized?** (Hidden trace means you can truncate without UX impact; returned trace means cancellation has to land cleanly mid-stream and you can't truncate aggressively.)
3. **Cancellation semantics.** Is "the user closed the tab" the same as "the user clicked stop"? Both should free GPU resources, but the first arrives as a TCP RST while the second is an explicit API call — these need different propagation paths.
4. **Per-user concurrency cap.** Is a single user allowed to fire 100 reasoning requests in parallel, or is there a per-user in-flight limit? (Without a cap, one user can pin half your fleet — common DoS vector with reasoning models.)
5. **Quality-vs-cost dial.** Is there a "fast / balanced / deep" mode the user picks, or is everything full-reasoning? (Affects whether you route to a non-reasoning fallback for short queries.)

**Reasonable assumptions to bake in if the interviewer waves off:**
- Smooth long-tail output distribution; size for it directly, no clever classification routing in v1
- Reasoning trace is streamed and visible to the user (the o1-style UX where you watch it think)
- Both cancellation paths must free GPU resources within 100ms of detection
- Per-user concurrent in-flight cap of 4 reasoning requests; soft limit, returns 429 on exceed
- Single mode in v1; in v2 a "fast" mode that routes to a 70B non-reasoning model for trivial queries

## 2. The right answer in one sentence

**A dedicated 70B-reasoning pool — physically isolated from any chat workload — sized by KV memory (not throughput), with first-class cancellation that frees KV pages within 100ms, hard per-request token budgets, SSE streaming of the reasoning trace from the first token, and EAGLE-3 spec decode (which pays off more on long decodes than short ones).**

The sentence that separates this from the bluff answer: **size the pool by KV memory, not by tokens/sec.** Reasoning workloads are the rare LLM serving case where KV-memory dominates everything else — at 30K-token p99 outputs, a single in-flight request can consume 5–10GB of KV cache; you run out of slots long before you run out of compute. The candidate who reaches for "QPS × tok/s / GPU throughput" math (the right answer for chat) and applies it here gets the GPU count off by 3–5×. The second senior-signal beat: **separate pool, no mixing with chat.** Continuous batching is throughput-optimal but fairness-blind — a 30K-token decode shares the batch with a 200-token chat reply for 60× longer, starving every chat request in the same slot.

## 3. The architecture (whiteboard)

```
                          Internet
                             │
                             ▼
                  ┌────────────────────────┐
                  │   Gateway              │   ─ TLS, auth (JWT)
                  │   (Envoy)              │   ─ per-user in-flight counter
                  │                        │     (Redis: user_id → count)
                  │                        │   ─ 429 if count >= 4
                  │                        │   ─ stamps request_id, OTel
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   Reasoning Router     │   ─ NOT shared with chat router
                  │   (custom — derived    │   ─ KV-aware: prefers replica
                  │   from vLLM Production │     with most free KV blocks
                  │   Stack)               │   ─ holds cancellation socket
                  │                        │     per in-flight request
                  └───────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ Replica1 │    │ Replica2 │ ...│ ReplicaN │   ─ vLLM 0.7+
       │ R1-70B   │    │ R1-70B   │    │ R1-70B   │   ─ TP=4 across
       │ FP8/MXFP4│    │ FP8/MXFP4│    │ FP8/MXFP4│     4×H100 OR
       │          │    │          │    │          │     1×MI300X (192GB)
       │ KV-budget│    │ KV-budget│    │ KV-budget│   ─ EAGLE-3 spec
       │ enforce  │    │ enforce  │    │ enforce  │     decode
       │ per-req  │    │ per-req  │    │ per-req  │   ─ cancellation hook
       └────┬─────┘    └────┬─────┘    └────┬─────┘     in scheduler
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                  ┌────────────────────────┐
                  │   Stream multiplexer   │   ─ SSE fanout per request
                  │   (per-replica)        │   ─ keepalive every 5s
                  │                        │   ─ on cancel: send EOS,
                  │                        │     close socket, signal
                  │                        │     scheduler to evict
                  └───────────┬────────────┘
                              │ SSE stream
                              ▼
                          to client

         ┌─────────────────────────────────────┐
         │  Control plane                      │
         │  ─ KEDA on TWO metrics:             │
         │     vllm:gpu_kv_cache_usage_perc    │
         │     vllm:num_requests_waiting        │
         │  ─ OTel GenAI spans per request     │
         │     incl. cancellation reason       │
         │  ─ Cost dashboard:                  │
         │     $/successful_completion         │
         │     $/cancelled_request (waste)      │
         │  ─ Hard kill: requests > 50K tokens │
         └─────────────────────────────────────┘
```

### The five-box mapping
- **Gateway:** Envoy with per-user in-flight counter in Redis. Returns 429 on exceed. Critical: this is enforced *before* the request hits the model — once it's in flight, killing it costs more than rejecting it.
- **Router:** A separate router from the chat router, sized for low-QPS / long-lifetime requests. Uses KV-cache-aware routing (free-block-count preferred), holds the cancellation socket per in-flight request so abort signals route to the right replica without a directory lookup.
- **Scheduler:** vLLM's continuous batcher inside each replica, with KV-budget-per-request enforcement on top — a request that exceeds its budget gets evicted, not allowed to grow unbounded.
- **Worker:** vLLM 0.7+, R1-70B class model, TP=4 on H100s OR single MI300X 192GB (the bandwidth wins for long decode). EAGLE-3 spec decode enabled.
- **Control plane:** KEDA scaling on `vllm:gpu_kv_cache_usage_perc` (the right metric for KV-bound workloads), OTel spans tagged with `cancellation_reason` (user / budget / timeout / error), cost dashboard separating successful tokens from wasted-on-cancel tokens.

**The senior signal:** drawing a *separate pool* from any chat workload, and explicitly naming KV-memory as the scaling dimension. The bluff answer ("we'd add an LLM service") leaves both differentiators on the floor.

## 4. The capacity math

This is the part that distinguishes someone who has actually served a reasoning model from someone who has only served chat. The math is fundamentally different.

```
Workload:
  20 QPS sustained. Output: mean 2000 tok, p99 30000 tok.
  Input ≈ 500 tok mean (modest — reasoning prompts are short, system prompts
  set up the thinking pattern).

LIFETIME math — what's actually in flight at any moment:
  Mean decode @ ~90 tok/s/req (R1-70B FP8 with EAGLE-3 spec decode, TP=4):
    mean lifetime = 2000 / 90 = ~22s
    p99 lifetime  = 30000 / 90 = ~333s ≈ 5.5 minutes

  In-flight requests (Little's Law: L = λ × W):
    mean in-flight = 20 QPS × 22s = 440 requests
    p99 in-flight  = 20 QPS × 333s = 6660 requests — but tail of the tail,
                     not every request hits this

  More usefully: weighted-average lifetime including the tail:
    E[lifetime] ≈ 0.99 × 22 + 0.01 × 200 (avg of tail) ≈ 23.8s
    Realistic in-flight = 20 × 23.8 = ~480 requests baseline,
    with surge to ~600-700 when several long ones overlap.

KV MEMORY math — the binding dimension:
  R1-70B (Llama-style arch, ~80 layers, head_dim 128, ~64 KV heads with GQA):
    Per-token KV (FP16): 2 × 80 × 64 × 128 × 2 bytes = 2.6 MB/token
    Per-token KV (FP8):  half of that = 1.3 MB/token

  At mean output 2000 tok per req:
    KV per req at completion = 2000 × 1.3 MB = 2.6 GB
    Average KV per in-flight req (linearly fills over lifetime) ≈ 1.3 GB

  At p99 output 30000 tok per req:
    KV per req = 30000 × 1.3 MB = 39 GB (!!) — single req can saturate a GPU
    This is why per-request KV budgets exist.

  Total KV memory needed for ~480 in-flight (avg-filled):
    480 × 1.3 GB = ~620 GB of KV memory across the fleet

GPU COUNT — sized by KV, not throughput:
  Per H100 80GB after weights (70B FP8 weights @ TP=4 means ~17.5 GB/GPU)
  and runtime overhead: ~55 GB KV per H100.
  TP=4 replica = 4 × 55 = 220 GB KV per replica.
  Replicas needed for 620 GB: 620 / 220 = 2.8 → 3 replicas baseline
  Plus headroom for p99 surge (in-flight peaks ~700, KV ~900 GB):
    900 / 220 = 4.1 → 5 replicas at peak
  Total: 5 × TP=4 = 20 H100s baseline, 24 H100s under p99 surge.

  Sanity check throughput-bound:
    decode aggregate = 20 QPS × 2000 tok = 40K tok/s
    per replica TP=4 @ batch 16 with spec decode: ~1500 tok/s × ~16 reqs
      = ~24K tok/s/replica
    needed: 40K / 24K = ~1.7 replicas — far less than KV-bound count.
  Conclusion: KV-bound, by ~3×. THIS is the load-bearing observation.

MI300X variant (the smart-money choice for this workload):
  Single MI300X = 192 GB HBM3e, 5.3 TB/s memory bandwidth (vs H100 ~3.3 TB/s).
  - Weights at TP=1: 70B FP8 ≈ 70 GB → fits with 120 GB left for KV
  - 120 GB KV per MI300X — same role as TP=4 H100s but on one GPU
  - Higher bandwidth = faster decode (decode is memory-bound on big models)
  - Estimated: ~5-6 MI300X for the same workload
  - Cost: comparable to 20×H100, often better (especially at decode shape)

GROWTH (100 QPS in 6 months):
  Mean in-flight = 100 × 24 = 2400 requests
  KV needed ≈ 3.1 TB
  → ~14 replicas of TP=4 H100 (56 GPUs) OR ~26 MI300X
  → mandates disaggregated prefill/decode at this scale (Topic 08)
```

### Cost-to-serve math, with cancellation waste

```
Successful completion cost (mean 2000 tok @ 90 tok/s, baseline replica busy):
  GPU-seconds per request = 22s × (1 GPU-of-TP=4 share) ≈ 5.5 H100-seconds
  At $4/H100-hr → $0.0061 per successful request

Cancelled request waste (user abandons at 20s of a 60s request):
  Pure waste: ~20 GPU-seconds × $4/3600 = $0.022 per cancellation
  At 15% cancellation rate (typical for reasoning UIs) and 20 QPS:
    3 cancels/s × $0.022 = $5,700/month of pure waste

THE 100ms cancellation propagation target buys us:
  Saved GPU time per cancel: ~5-10s of decode that would have continued
  Saved waste: 60-70% of the $5,700 → ~$3,800/month saved
  Worth engineering for.
```

## 5. The hard parts — what actually breaks

### 5a. Pool isolation — why mixing kills

The single most-mistaken design choice for reasoning models is sharing a pool with chat. Here's the failure in concrete terms:

```
Shared continuous batch (BAD):
  Batch contains: 1 reasoning req (will decode 2000 tok) + 15 chat reqs (200 tok each)
  Continuous batching: each step runs ALL active reqs together
  Chat reqs finish at step 200 → batch slot frees → new chat req joins
  Reasoning req still running at step 2000 → it's been in batch with
    10 GENERATIONS of chat reqs by now
  Throughput per chat req: degraded by (long_req_kv_pressure /
    available_kv) → 30-50% slower than dedicated pool
  Tail latency for chat: dominated by KV pressure from reasoning, not by
    its own decode steps

Separate pools (GOOD):
  Reasoning pool sized by KV for long decodes; KV budget per request enforced
  Chat pool sized by throughput for short decodes; no long-decode contamination
  Each pool's batch fills with peers of similar lifetime — batching efficiency
  recovered
```

There's a published version of this lesson in DeepSeek's R1 deployment notes and in vLLM issue tracker discussions on "long-tail latency under mixed workloads." The fix is structural, not configurable: separate fleets.

### 5b. Cancellation propagation as a first-class path

Cancellation in a reasoning model is *the* thing most candidates forget. Concrete design:

```python
# Router holds cancellation socket per in-flight request
class ReasoningRouter:
    def __init__(self):
        self.in_flight: dict[str, ReplicaConn] = {}  # request_id → replica

    async def submit(self, req: ReasoningRequest, cancel_signal: asyncio.Event):
        replica = self.pick_replica_kv_aware(req)
        self.in_flight[req.id] = replica

        # Launch the generation
        gen_task = asyncio.create_task(replica.generate_stream(req))

        # Watch for cancellation in parallel
        cancel_task = asyncio.create_task(cancel_signal.wait())

        done, pending = await asyncio.wait(
            [gen_task, cancel_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_task in done and gen_task not in done:
            # Cancellation arrived first — propagate immediately
            await replica.abort(req.id)            # < 10ms RPC
            gen_task.cancel()                       # cancel coroutine
            # vLLM's scheduler hook frees KV pages on next iteration step
            # Net cancellation-to-KV-free latency: < 100ms

        del self.in_flight[req.id]
```

In the vLLM scheduler, the abort path is:

```python
# inside vLLM engine — already exists, but we tune the iteration cadence
class Scheduler:
    def step(self):
        # Check abort queue at every step (every ~15ms in continuous batch)
        for req_id in self.abort_queue.drain():
            req = self.running.pop(req_id)
            self.block_manager.free(req)           # frees KV blocks
            # Emit final SSE token to drain the stream
            req.output_queue.put(AbortedToken())
```

vLLM's `step` interval is typically 10–20ms. Combined with router RPC time, end-to-end cancellation lands KV-free in <100ms — the SLO we set up front.

**Three propagation triggers, all routed through the same abort path:**
- Explicit user "stop" button → API DELETE → router → replica
- Browser tab close → TCP RST on SSE socket → router detects via stream handler → abort
- Token budget exceeded (server-side) → scheduler self-aborts the req

### 5c. KV-budget-per-request enforcement

A reasoning model that doesn't converge can decode forever. This is a cost-runaway vector — left unchecked, one bad prompt costs you $5+ in compute. The fix:

```python
# Per-request token budget enforced inside the scheduler
class TokenBudgetEnforcer:
    HARD_LIMIT = 50_000   # absolute cap on output tokens
    SOFT_LIMIT = 30_000   # warning, model can self-terminate

    def on_token(self, req: Request, token_id: int):
        if req.output_len >= self.HARD_LIMIT:
            req.abort(reason="token_budget_exceeded")
            log.warning(f"req {req.id} hit hard limit; user={req.user_id}")
            return AbortedToken()

        if req.output_len == self.SOFT_LIMIT:
            # Inject a "wrap up" system message via prefix or stop signal
            req.set_stop_pattern("</thinking>")
```

This is non-negotiable. Reasoning models without budget enforcement are a P0 cost incident waiting to happen.

### 5d. EAGLE-3 spec decode pays MORE here

Speculative decoding's effective speedup is `(γ × acceptance_rate) / (γ × cost_ratio + 1)` where γ is draft length. The fixed cost of the draft model is amortized over the full decode length — long decodes amortize better.

```
Chat workload (300 tok decode):
  EAGLE-3 with γ=5, acc=0.7 → effective speedup ~1.6×

Reasoning workload (2000 tok decode):
  Same EAGLE-3 setup → ~1.7× speedup
  But spec decode reduces wall time by the SAME 40%, applied to a much
  larger absolute number — saves 13s instead of 1.7s per request
```

EAGLE-3 specifically (vs. older Medusa or vanilla spec) gets used because acceptance rates stay above 0.65 on reasoning traces where the model is largely re-stating prior context. Setup:

```python
{
  "model": "deepseek-ai/deepseek-r1-distill-llama-70b",
  "quantization": "fp8",
  "tensor_parallel_size": 4,
  "speculative_config": {
    "model": "eagle3-r1-70b-draft",
    "num_speculative_tokens": 5,
    "draft_tensor_parallel_size": 1
  },
  "enable_prefix_caching": true,       # system-prompt prefix-caching
  "gpu_memory_utilization": 0.90,
}
```

### 5e. Streaming the trace from first token

UX bargain: the user accepts 60s wait IF they see incremental progress. The model must stream the first reasoning token within ~1s, and continue streaming until done. SSE is the wire format; the router holds one socket per in-flight request.

```
Server timeline (typical successful 5s request):
  t=0       request hits gateway
  t=15ms    router picks replica, opens SSE socket to client
  t=140ms   prefill complete, first decode step
  t=160ms   first token sent over SSE — user sees activity
  t=5000ms  EOS, stream closes
```

If the user closes the tab at t=20s on a longer request, the SSE socket dies; router detects within one TCP keepalive window (~5s if no explicit close, ~50ms on graceful close) and triggers the abort path.

## 6. The break-it list

| Failure | What happens | Your mitigation |
|---|---|---|
| One user submits 100 reasoning requests in parallel | Pool fills with one user's requests; everyone else queues | Per-user in-flight cap at gateway (Redis counter); 429 on exceed |
| Prompt loops the model (no convergence) | Request decodes for 5+ minutes; cost explodes | Hard 50K-token budget enforced in scheduler; soft 30K with wrap-up signal |
| KV pressure exceeds budget mid-batch (request grows past forecast) | vLLM evicts a request → unrecoverable abort | Per-request KV reservation at admission time; if reservation can't be met, queue or reject (not admit-and-evict) |
| Browser tab close not propagated → orphaned generation | Wasted GPU time, KV held forever | SSE keepalive every 5s; on broken pipe, router triggers abort path within 100ms |
| Spec decode draft regresses (acceptance rate drops below 0.4) | Decode becomes slower than no-spec | Continuous acceptance-rate monitor; auto-disable spec at <0.5 acceptance until investigated |
| Reasoning trace contains user PII that gets logged | Compliance event | OTel span captures only token counts and metadata, NEVER trace content; trace content only in user's SSE stream |
| Single replica loses 1 of 4 GPUs (TP=4 dies) | Replica goes hard down; in-flight requests lost | TP=4 → degrade to TP=2 not possible mid-flight; replica marked unhealthy, drains queued; in-flight requests get 503 + auto-retry on healthy replica |
| Long request blocks autoscale-down | KEDA can't drain replica until p99 lifetime passes (5+ min) | Graceful drain window = 6 min; if exceeded, force-cancel oldest in-flight requests, return retriable error |
| Cost dashboard shows 30% spend on cancelled requests | Real money on the floor | Investigate UX — is the "stop" button too prominent? Tune timeout defaults; consider partial-result return on cancel |
| New R1 model version regresses on hard prompts | Quality drops in production | Per-prompt-class eval gate at release time (math / code / open-ended); canary 5% traffic for 48h before full rollout |
| MI300X variant — ROCm vLLM build lags upstream | Stuck on older vLLM | Maintain dual-target CI (CUDA + ROCm) for the engine repo; AMD support engineer on retainer if the workload is big enough |
| User opens stream, then their network drops mid-decode | TCP socket hangs in CLOSE_WAIT | Aggressive TCP keepalive (10s) + SSE app-level ping; on detected dead socket, abort within 100ms |

## 7. What changes at 10× scale

```
At 100 QPS sustained (10× of current 20 QPS, the 6-month target):

Disaggregation (now mandatory):
  - Prefill pool: small, fast — B200s or H200s with strong compute
    (prefill is compute-bound on R1 with ~500-token inputs)
  - Decode pool: large — MI300X 192GB with HBM3e bandwidth
    (decode is memory-bandwidth-bound; bandwidth × KV capacity wins)
  - Dynamo or llm-d as orchestrator (Level 5 Topic 09)
  - KV transfer between pools via NVLink/InfiniBand RDMA — < 5ms
  - Pool ratio: ~1:5 prefill:decode by GPU count for this output shape

KV management:
  - LMCache (Topic 12) for cross-replica system-prompt prefix-caching
  - At this scale, system prompts are stable; prefix cache hit ~95%
  - Saves ~50ms of prefill per request, frees compute for new prefills

Adaptive abort / quality-vs-cost routing:
  - Cheap classifier at admission: "is this a 'what's 2+2' query or a real
    reasoning task?" → trivial queries → 70B non-reasoning model (10× cheaper)
  - Mid-stream confidence monitor: if reasoning model hasn't converged on
    a confident answer pattern by 5K tokens, abort and fall back to
    non-reasoning summary path
  - User-facing: "this seems straightforward — answering in fast mode" UX

Per-user enforcement:
  - In-flight cap moves from gateway-only to per-tenant queue with WFQ
    (Level 7 Topic 07)
  - Cost-budget-per-user-per-day enforced at gateway
  - Premium tiers get higher concurrency and priority routing

Reliability:
  - Multi-region active-active; user's in-flight requests pinned to their region
  - On region failure, in-flight requests are LOST (no cross-region KV migration —
    too expensive); user retries land in healthy region
  - Per-region capacity sized for full failover load (2× normal) on at least
    one region

Team-shape:
  - Dedicated reasoning-platform engineer (separate from chat-platform)
  - The workloads are different enough that one engineer owning both compromises
    both
```

**The axis of change:** at 100 QPS, the design transitions from "one pool with smart enforcement" to **disaggregated prefill/decode plus adaptive routing**. The KV-bound math gets worse linearly (1.3 TB → 3+ TB), but disagg lets you scale the decode side independently — and MI300X's 192 GB HBM is purpose-built for this shape.

## 8. The 30-second summary you give the panel

> "Reasoning workloads break the chat-serving playbook in three places: KV memory dominates throughput, long decodes starve short ones in shared batches, and cancellation matters as a first-class cost-control mechanism. I'd run a dedicated R1-70B FP8 pool — physically separated from any chat workload — sized at ~20 H100s in TP=4 replicas, OR ~6 MI300X using the 192 GB HBM3e for KV. The math is KV-bound at ~620 GB needed across the fleet, not throughput-bound. Cancellation is engineered to free KV pages within 100ms via a router-held abort socket. Hard 50K-token budget per request kills the cost-runaway case. EAGLE-3 spec decode pays even more here because the long decodes amortize the draft cost — ~1.7× speedup, ~13s saved per mean request. SSE streaming from first token makes the 60s wait acceptable UX. At 100 QPS I'd disaggregate prefill onto B200s and decode onto MI300X pool, add adaptive abort for non-converging traces, and per-user concurrent caps to prevent one user pinning half the fleet. The single most-mistaken design here is sharing the pool with chat — that destroys both."

## What this prompt is really testing

- **Recognizing that reasoning is a different workload shape** — KV-bound, not throughput-bound; long-tail decode, not short decode. The candidate who applies chat-serving math gets the answer wrong by 3×
- **Pool isolation as a structural choice** (Level 7 Topic 15) — mixing pools is the most common production mistake, and the candidate either sees it or doesn't
- **Cancellation as first-class concern** — the seniority signal. Most candidates forget cancellation exists; senior candidates name a latency target (100ms) and design for it
- **KV memory budgeting per request** — the cost-control discipline that separates someone who's been on-call for a reasoning model from someone who's read about them
- **Disaggregated prefill/decode** (Level 5 Topic 08) — knowing when to reach for it (long-decode workloads, especially with bandwidth-limited GPUs)
- **MI300X awareness** — the practitioner knows that 192 GB HBM3e wins on KV-bound decode-heavy workloads; the textbook reader still defaults to H100
- **Spec decode tuning** (Level 5 Topic 12) — knowing why EAGLE-3 pays MORE here, not less
- **Migration thinking** — knowing the single-pool design holds to ~50 QPS, then disagg becomes mandatory, is the seniority signal

## References

- [Level 7 Topic 15 — reasoning-aware serving](../../../level-7-ml-platform/15-reasoning-aware-serving/)
- [Level 7 Topic 07 — fairness / WFQ](../../../level-7-ml-platform/07-fairness/)
- [Level 7 Topic 08 — backpressure](../../../level-7-ml-platform/08-backpressure/)
- [Level 7 Topic 12 — KV-tiering / LMCache](../../../level-7-ml-platform/12-kv-tiering/)
- [Level 7 Topic 13 — cost-economics](../../../level-7-ml-platform/13-cost/)
- [Level 5 Topic 08 — disaggregated prefill/decode](../../../level-5-production-engines/08-disaggregated-prefill-decode/)
- [Level 5 Topic 09 — Dynamo + llm-d](../../../level-5-production-engines/09-dynamo-llmd/)
- [Level 5 Topic 12 — production spec decode (EAGLE-3)](../../../level-5-production-engines/12-spec-decode-prod/)
- [Level 4 — paged KV / FP8](../../../level-4-llm-optimization/)
- DeepSeek-R1 technical report (2025) — reasoning model serving observations
- EAGLE / EAGLE-3 papers (Li et al., 2024 / 2025) — spec decode for long decodes
- Kiely *Inference Engineering* Ch 6 §6.2–6.4 — long-decode workloads and cancellation
- vLLM scheduler abort path: `vllm/core/scheduler.py` — the load-bearing code
