# Prompt 02 — Worked Solution

> The "compounding optimization stack" prompt. At 10K QPS on a 70B chat model, every Level 4 technique pays back in tens of millions per year. This is where the optimization-stack-composition muscle gets tested.

## 1. Clarifying questions (the first 3 minutes)

1. **Multi-turn or single-shot?** Chat almost certainly multi-turn — that's the entire case for RadixAttention / prefix caching. Confirm; this is the highest-leverage assumption.
2. **Conversation history length distribution.** Mean is 600 tokens — is that *first message* or *with history*? If history grows to 4K-32K tokens by turn 10, KV pressure becomes the binding constraint.
3. **End-to-end p99 includes network RTT or just server?** 2s server-side leaves comfortable budget; 2s end-to-end with EU users on US servers is much tighter.
4. **First-token vs complete-response SLO.** Chatbots are TTFT-sensitive — users feel <500ms TTFT as "instant" regardless of total. Optimize for TTFT, not full-response p99.
5. **EU residency / GDPR.** Strict residency or "best effort"? If strict, EU users → EU pool, no cross-region KV federation for their conversations.
6. **Cost vs quality tradeoff.** Are we allowed to use FP8 / Int4? Quality regression > 1% acceptable, > 2%? (This is the gate that lets us collect ~30-50% savings.)

**Assumptions baked in:** multi-turn with history growing to ~4K tokens by turn 10; p99 < 2s server-side; TTFT < 400ms is the *real* SLO that determines UX; FP8 acceptable if `lm-eval-harness` regression < 1%; EU active-active with EU-pinned KV.

## 2. The right answer in one sentence

**SGLang with RadixAttention for the chat-shape workload (prefix-tree KV reuse beats vLLM here by 2-3× on hit rate) + Dynamo's disaggregated prefill (B200, small pool) + decode (MI300X, massive pool) + FP8 quantization + EAGLE-3 speculative decoding + KV-cache-aware routing + owned H100/B200/MI300X fleet across US-East / US-West / EU regions. Every optimization compounds; the stack is mandatory at this scale.**

The senior signal here: **SGLang over vLLM** for the multi-turn chat case, with a specific reason (RadixAttention's prefix-tree gives higher cache reuse on growing-history workloads). Most candidates default to vLLM out of habit.

## 3. The architecture (whiteboard)

```
              ┌──────────────────────────────────────┐
              │     CloudFront / Cloudflare          │  Anycast L3, then
              │     (geo-routing, TLS termination)   │  region pin
              └──────────────────┬───────────────────┘
                                 │
       ┌─────────────────────────┼────────────────────────┐
       ▼                         ▼                        ▼
  ┌─────────┐              ┌─────────┐              ┌─────────┐
  │ US-East │              │ US-West │              │   EU    │
  │ region  │              │ region  │              │ region  │
  └────┬────┘              └────┬────┘              └────┬────┘
       │  (active-active, ~33% of US traffic each;       │
       │   EU is 30% of global; APAC + India added       │
       │   in Phase 2)                                   │
       ▼                                                  │
  ┌─────────────────────────────────────────────────────┐ │
  │   Envoy AI Gateway                                  │ │
  │   ─ TLS, mTLS to backend                            │ │
  │   ─ rate-limit per-account (free vs paid tiers)     │ │
  │   ─ OpenAI-compatible /v1/chat/completions          │ │
  │   ─ session affinity hint (conversation-id header)  │ │
  └─────────────────────────┬───────────────────────────┘ │
                            │                             │
                            ▼                             │
  ┌─────────────────────────────────────────────────────┐ │
  │   dynamo-router (KV-aware)                          │ │
  │   ─ hashes by conversation_id → sticky to replica   │ │
  │     that owns this conversation's KV                │ │
  │   ─ on cache miss (cold-start or rebalance),         │ │
  │     routes to least-loaded replica                  │ │
  │   ─ free-tier vs paid-tier priority queueing        │ │
  └─────────────────────────┬───────────────────────────┘ │
                            │                             │
            ┌───────────────┼───────────────────┐         │
            ▼               ▼                   ▼         │
       ┌──────────┐   ┌──────────────┐   ┌──────────────┐ │
       │ Prefill  │   │  Decode pool │   │ EAGLE-3 draft│ │
       │ pool     │   │  (SGLang +   │   │ model worker │ │
       │ (SGLang  │   │  RadixAtten- │   │ (1B Llama-3- │ │
       │  on B200,│   │  tion +      │   │  1B-Instruct,│ │
       │ ~small)  │   │  FP8 weights,│   │ same family) │ │
       │          │   │  MI300X)     │   │              │ │
       └────┬─────┘   └──────────────┘   └──────────────┘ │
            │   │ NIXL transfer of KV state                │
            └───┘                                          │
                            │                             │
                            ▼                             │
              ┌──────────────────────────────┐            │
              │  LMCache regional KV tier    │            │
              │  ─ HBM (current convos)      │            │
              │  ─ DRAM (recent dormant      │            │
              │    conversations, 1h TTL)    │            │
              │  ─ NVMe (long-tail, 24h TTL) │            │
              └──────────────────────────────┘            │
                            │                             │
                            ▼ (OTel spans)                │
              ┌──────────────────────────────┐            │
              │ Control plane                │◄───────────┘
              │ ─ KEDA on num_requests_      │  (single
              │   waiting + decode_queue     │   global)
              │ ─ Prometheus + Grafana       │
              │ ─ Argo CD + per-model eval   │
              │   gates                      │
              └──────────────────────────────┘
```

### Five-box mapping

- **Gateway:** Envoy AI Gateway. Per-account rate-limiting (free tier: 20 req/min, paid: 600 req/min, enterprise: unlimited within fair-share). Session affinity hint via `conversation-id` header so the router can do sticky routing.
- **Router:** dynamo-router hashing by `conversation_id`. Within a region, this gives ~90% KV-cache hit on turn 2+ of any conversation. Cross-region is handled by region-pinning at the gateway; we don't try to route US-East users to EU.
- **Scheduler:** SGLang's continuous batcher inside the engine; per-tier WFQ (free vs paid) at the router.
- **Worker:** Disaggregated. Prefill on B200 (small pool, ~5% of total fleet), decode on MI300X (huge pool, 90% of cost). EAGLE-3 draft model co-located with decode workers for spec decode.
- **Control plane:** KEDA on `vllm:num_requests_waiting` and `sgl:decode_queue_depth`; Prometheus + Grafana + Tempo; Argo CD; eval gates that block any model deploy regressing >1% on `lm-eval-harness`.

## 4. Capacity math

```
Mean throughput required (10K QPS peak):
  input_tok/s  = 10,000 × 600 = 6.0M tok/s
  output_tok/s = 10,000 × 200 = 2.0M tok/s

But — multi-turn — the *effective* input per turn after prefix cache is much less:

  Assume 70% prefix-cache hit rate on multi-turn chat with RadixAttention.
  Effective prefill work = 6.0M × 0.30 = 1.8M tok/s
  Decode work unchanged   = 2.0M tok/s (decode always pays full cost)

Engine perf (Llama-3-70B FP8 on the right hardware):
  prefill on B200 (FP8): 65,000 tok/s/GPU
  decode  on MI300X:      1,500 tok/s/GPU
  with EAGLE-3: 2,100 effective tok/s/GPU
      (~1.4× — the realistic high-batch gain for a dense 70B at 10K QPS;
       see §5.2. The textbook 2× is a concurrency-1 figure and does NOT
       hold here. Deep multi-turn histories can push toward ~2× because
       long-context decode stays bandwidth-bound — quote that conditionally.)
  concurrent decode slots: 64/GPU

GPUs needed at peak:
  prefill = 1.8M / (65K × 0.70)   = 39.5 → 40 B200
  decode  = 2.0M / (2.1K × 0.70)  = 1,360 → 1,360 MI300X
  concurrency = (10K × 4s) / 64    = 625 → 625 MI300X (non-binding)
  binding = decode @ 1,360

  × 1.3 (p99 headroom)      = 1,768
  + 2   (N+2 redundancy)    = 1,770
  × 1.15 (warm pool)        = 2,036 MI300X per global fleet
  Plus 40 B200 × 1.5 = 60 B200 prefill (rounded with headroom)

Total: ~2,036 MI300X + 60 B200 + a few hundred small GPUs for EAGLE-3 draft worker pool.
At 3 regions (US-East/West, EU): ~680 MI300X per region.

Cost (optimized fleet):
  MI300X: ~$2.10/hr blended (committed + on-demand mix)
  B200:   ~$5.50/hr blended
  Daily compute = 2,036 × 24 × 2.10 + 60 × 24 × 5.50
               = $103K + $8K = $111K/day ≈ ~$40M/year

Token economics:
  Tokens/year = 10K QPS × (600+200) × 86400 × 365 = ~250 billion tokens/year
              = 250,000 Mtok/year
  $/Mtok (optimized) = $40M / 250,000 = $0.16/Mtok of compute
      (note: $/Mtok here is amortized fleet cost ÷ tokens; it's well below the
       ~$0.72 marginal serving cost in the table below because the table starts
       from a naive single-replica baseline, not an amortized committed-use fleet.)

  Run the SAME sizing with NONE of the optimizations (BF16, no spec decode, no
  prefix cache, no disagg) and the decode pool balloons to ~7,500 MI300X →
  ~$160M/year. THAT delta — ~$40M optimized vs ~$160M naive, ~$120M/year — is
  the whole point. Every Level 4 technique below buys back part of that gap.
```

### The compound optimization stack — where the $$ comes from

Starting from a naive baseline (vLLM, BF16, no spec decode, no prefix cache, no disagg), each optimization peels off cost:

```
Optimization                          Effect              $/Mtok cut
──────────────────────────────────────────────────────────────────────
Baseline (vLLM BF16, no opt)          ─                   $4.00
+ FP8 quantization                    ~30% throughput     → $2.80   (-30%)
+ SGLang RadixAttention               prefix reuse on     → $1.95   (-30%)
                                      multi-turn chat
+ EAGLE-3 spec decode (~1.4× @ batch) ~30% fewer decode   → $1.45   (-26%)
                                      GPUs (NOT 2× — §5.2)
+ Disagg prefill/decode               ~20% more efficient → $1.16   (-20%)
+ KV-cache-aware routing              compounds w/ Radix  → $1.05   (-9%)
─────────────────────────────────────────────────────────────────────
FINAL marginal $/Mtok                                     ~$1.05
SAVINGS vs naive baseline:  ~74%

GPUs needed drops from naive ~7,500 MI300X to optimized ~2,036 MI300X.
At committed-use amortized fleet pricing: ~$160M/year naive → ~$40M/year
optimized = ~$120M/year saved by the optimization stack.
```

(The per-Mtok column and the amortized-fleet figure measure different things —
marginal serving cost vs. committed-fleet cost ÷ annual tokens — so don't expect
them to multiply out to the same number. State which one you mean in the room.)

**This is the single most important takeaway from the prompt.** A senior engineer wiring the optimization stack into production is making a roughly $120M/year decision. The interviewer wants you to verbalize this math live — *and* to be honest that EAGLE-3 contributes ~1.3–1.5× at this batch regime, not the headline 2×.

## 5. The hard parts

### 5.1 Why SGLang over vLLM specifically for *this* workload

vLLM and SGLang are both excellent, and as of 2026 **both** do automatic prefix
caching well — get the framing right or an interviewer who runs vLLM will catch you.

```
vLLM V1 automatic prefix caching (APC):
  ON BY DEFAULT in the V1 engine (disable only via --no-enable-prefix-caching)
  Block-hash-keyed: each KV block hashed by its tokens + the preceding block
  Multi-turn chat: turn N reuses the cached blocks of turns 1..N-1 →
  genuine incremental hits, NOT exact-match-only. This is a real, default feature.

SGLang RadixAttention:
  Stores KV state as a radix tree of token sequences (token-level, not block-level)
  Matches the longest shared prefix path in the tree
  Edge on prefix-heavy / high-reuse workloads (deep multi-turn, shared system
  prompts, RAG): finer-grained sharing + eviction tuned for it.
```

So the honest 2026 picture is **workload-dependent, not a blanket win.** On standard
mixed throughput SGLang leads vLLM by ~25-30% (≈16K vs 12.5K tok/s, 8B/H100); the gap
shrinks toward zero on unique-prompt batch jobs and only balloons to ~6× on the very
highest-cache-hit RAG / deep-multi-turn cases. For everyday multi-turn chat the gap is
real but modest. The senior signal is **picking SGLang for the prefix-heavy chat shape
*and* knowing vLLM V1 closes most of the gap with default APC + KV-aware routing** —
not parroting a "vLLM can't prefix-cache" myth that was never true of V1.

That said — SGLang has fewer features than vLLM at any given moment (TensorRT-LLM features lag vLLM lag SGLang in a rolling 6-month cycle). At this scale you might run *both*, with SGLang for the prefix-heavy chat workload and vLLM for any single-shot tasks.

### 5.2 EAGLE-3 spec decode (Level 5 Topic 12) — why it pays back

EAGLE-3 uses a small draft model (~1B params) to propose multiple tokens speculatively; the target 70B model verifies them in a single forward pass. EAGLE-3 is real and supported in vLLM V1 (set `method: eagle3` in `speculative_config`).

**The number that matters — and where most candidates overclaim.** The headline "~2×" is a *low-concurrency / concurrency-1* figure (vLLM/Red Hat measure ~1.6–1.8× latency reduction at low request rates; the EAGLE-3 paper's 4–6× is temperature-0, batch-size-1 academic benchmarking). **At the high effective batch sizes a 10K-QPS service runs, a dense 70B is compute-bound, and the decode gain shrinks to ~1.2–1.5× — and can even go negative if the draft pass isn't earning its keep.** So you do *not* cleanly "halve the decode pool" at this scale. The honest planning number is ~1.3–1.5× decode throughput → ~25–35% fewer decode GPUs, not 50%.

There's one rescue specific to *this* workload: long conversation histories keep decode memory-bandwidth-bound even at large batch (KV-cache loading dominates), which restores closer to ~2× — so for deep multi-turn chat the gain is at the better end of the range. Quote that conditionally, not as a blanket 2×.

Cost: a tiny draft-model worker pool (1–2% of the decode pool size) + scheduler complexity.

**Caveat (the senior signal):** acceptance rate AND batch regime both matter. Production needs continuous monitoring — if the rolling acceptance × batch-adjusted speedup drops below ~1.2×, spec decode is hurting you (the draft forward pass costs more than the savings). Auto-disable on that signal.

### 5.3 KV pressure as conversations grow

Standard 70B + FP8 = ~80GB weight + ~10GB activations. KV cache per token (16 layers × 8 KV heads × 128 head dim × 1 byte FP8) ≈ ~16KB/token. A turn-10 conversation at 4K tokens = ~65MB of KV.

```
Per MI300X (192GB HBM3e):
  Weights:    80GB
  Activations: 10GB
  Available for KV: ~100GB
  → ~1,500 concurrent conversations at turn 10 per GPU

At 10K QPS, ~40K concurrent conversations across the fleet.
Per-GPU concurrency capacity is ~1,500 → need ~30 decode GPUs just by KV math.
This is well below the throughput-bound ~1,360; throughput binds, KV does not.

BUT — if conversation length grows (some users at turn 50, 4K tokens history)
the KV memory math can bind. LMCache tiering kicks in:
  Active turns: KV in HBM
  Dormant 1-min conversations: KV swapped to DRAM (8× cheaper)
  Dormant 1-hour: NVMe
  Brought back to HBM when next turn arrives (~10ms overhead, hidden under network)
```

### 5.4 Cold start during traffic burst

Standard cold start on Llama-3-70B-FP8: ~60s naive, ~25s with Run:ai Model Streamer (Topic 11). At 10K QPS, every cold start that takes 25s = ~250K requests that hit a slower replica or queue. This must be hidden:

```
Mitigations:
  1. Warm pool sized to absorb 2× normal traffic burst without cold-start (warm
     pool overhead is ~15% of fleet — built into capacity math above)
  2. KEDA fires on `num_requests_waiting > 50` *ahead* of SLO breach;
     reaction window is 30s, so scale-up completes BEFORE p99 jumps
  3. Pre-pull image via DaemonSet on every node (eliminates 5s pull time)
  4. Tensorized weights checkpoint format (loads in ~6s on B200)
  5. CRIU-style checkpoint of the warmed-up engine (Modal's trick); cuts 25s
     to ~8s if you own the snapshotting infrastructure (probably not at our
     scale unless we build it)
```

### 5.5 Multi-region routing & failover

EU customers route to EU region (GDPR + lower latency). US customers split 50/50 US-East / US-West. Failover: if EU region degrades, EU traffic spills to US-East (latency penalty ~100ms but service stays up). Cross-region failover takes 2-4 min via DNS TTL; we accept that for a partial-region outage.

The hard part: **what about the KV cache for EU conversations during failover?** Option A: lose the conversation context (user has to re-explain). Option B: replicate KV to a fallback region (expensive). For a chatbot we choose A — the conversation degrades gracefully but service stays up.

## 6. Break-it list

| Failure | What happens | Mitigation |
|---|---|---|
| EAGLE-3 acceptance drops below 1.8× | Spec decode now *hurts* throughput | Monitor accept rate; auto-disable spec decode if rolling 5-min avg < 1.8×; alert |
| Free-tier abuser DDoS | Noisy neighbor; fairness breaks | Per-account token bucket at Envoy; concurrent-request cap; auto-promote suspicious accounts to a "throttled" tier |
| Region outage (full EU down) | 30% of users see degraded latency | DNS-failover to US-East within 4 min; users lose KV context (gracefully); on recovery, fail back |
| Single MI300X dies under load | Lose ~0.07% capacity; conversations on that GPU lose KV | Routing re-pins on next request to a peer replica; conversation re-prefills from full history (paid by user wait of ~2s once); replacement GPU provisioned in 4 min |
| Bad model promote (FP8 quality regression) | All chat quality drops | Canary at 1% → 10% → 100% over 72h; auto-rollback on user-flagged-response rate > 2× baseline |
| RadixAttention memory leak | KV tree grows unbounded | Periodic LRU eviction; max tree size enforced; alert if eviction rate spikes (signals leak vs. organic growth) |
| KEDA reaction lag | Cold-start avalanche during a 10× burst | Pre-emptive scale-up on calendar-known events (product launches, news cycles); ML-based traffic predictor 30 min ahead |
| Cross-region KV federation gets out of sync | Cache misses temporarily 2× higher | Cache layer is best-effort eventual; misses just re-prefill; alert only if persistent > 30 min |
| New CVE in SGLang | Patch the fleet without downtime | Engine version per deployment; rolling update by region (one region at a time, 6h windows); canary 5% for 24h before full |
| EU customer's compliance auditor asks for data lineage | Need request-tracing | Every OTel span tagged with region, tenant, model, conversation_id; queryable for 13 months |

## 7. What changes at 100K QPS

At 10× the design, things stop scaling linearly:

**Custom kernels.** A 2% efficiency gain at this scale is $80M/year. The sibling `compiler-and-kernels/` track stops being aspirational and becomes a dedicated team contributing Triton/CUTLASS kernels back to SGLang/vLLM for our specific workload shapes (long-context decode, prefix-cache-heavy prefill).

**Co-design with the chip vendor.** Character.ai famously did this in 2024 with NVIDIA on the H200 release. At 100K QPS, your workload is large enough that the next-gen silicon roadmap can be shaped by your asks (longer NVLink reach, more HBM, custom tensor formats).

**Multi-model decoding.** Tier the chat by complexity: route simple turns to a fine-tuned 8B model, hard turns to the 70B. Detection model (1B classifier) decides. ~40% of turns are simple → ~40% cost reduction on those.

**Reasoning model offshoots.** Some chat turns are reasoning-shaped (the user says "think carefully about X"). Route those to a dedicated reasoning pool (Prompt 06) with different SLO and pricing.

**Real-time RLHF.** Continuous fine-tuning from user feedback signal (thumbs-up/thumbs-down, conversation continuation rate). vLLM/SGLang as rollout backend for verl/OpenRLHF (Level 6 Topic 15). New LoRA pushed daily.

**Multi-fabric networking.** InfiniBand within rack, NVLink 5 within node, Ultra Ethernet for cross-DC. The networking topology gets engineered.

**Org-shape change.** Three platform teams (one per region) + a kernels team + a chat-product team + an RLHF team. The chat workload becomes its own division.

## 8. The 30-second summary

> "For 10K QPS multi-turn chat on 70B, the optimization stack is mandatory and compounds: SGLang for RadixAttention (token-level prefix sharing edges out vLLM on the prefix-heavy chat shape, though vLLM V1's default APC closes most of the gap), FP8 quantization, EAGLE-3 spec decode, disaggregated prefill on B200 and decode on MI300X. Without this stack the fleet costs ~$160M/year; with it, ~$40M/year. Architecture is three active-active regions, Envoy AI Gateway with per-account rate limits, dynamo-router with KV-aware sticky routing per conversation_id, LMCache for HBM→DRAM→NVMe tiering on dormant conversations. The critical platform investment is monitoring EAGLE-3 acceptance rate live, because at our high effective batch sizes the spec-decode gain is closer to ~1.3-1.5× than the textbook 2×, and if acceptance drops too far the draft pass costs more than it saves. At 100× scale, custom kernels and co-design with the silicon vendor become the leverage."

## What this prompt is really testing

- **Optimization stack composition.** Naming each technique with its specific $/Mtok contribution. Most candidates can name them; few can stack them and compute the compound saving.
- **SGLang over vLLM specifically** — the workload-shape judgment.
- **Quantifying acceptance rate as a continuous monitor** — the senior signal that you understand spec decode isn't fire-and-forget.
- **The 82% savings story** — turning Level 4 into a board-deck-ready value prop.
- **KV memory binding awareness** — knowing when throughput binds vs. when KV binds.
- **At 10×, naming axes of change** — not "more GPUs," but custom kernels, vendor co-design, multi-model tiering.

## References

- [Level 4 Topic 02 — fp8-and-nvfp4](../../../level-4-llm-optimization/02-fp8-and-nvfp4/)
- [Level 4 Topic 13 — speculative-decoding](../../../level-4-llm-optimization/13-speculative-decoding/) + [Topic 17 — spec-decode-systems](../../../level-4-llm-optimization/17-spec-decode-systems/)
- [Level 4 Topic 10 — kv-cache-paged](../../../level-4-llm-optimization/10-kv-cache-paged/) + [11 — kv-cache-eviction](../../../level-4-llm-optimization/11-kv-cache-eviction/)
- [Level 5 Topic 03 — sglang-and-radixattention](../../../level-5-production-engines/03-sglang-and-radixattention/)
- [Level 5 Topic 08 — disaggregated-inference](../../../level-5-production-engines/08-disaggregated-inference/) + [09 — dynamo-and-llmd](../../../level-5-production-engines/09-dynamo-and-llmd/)
- [Level 5 Topic 12 — speculative-decoding-in-prod](../../../level-5-production-engines/12-speculative-decoding-in-prod/)
- [Level 7 Topic 06 — inference-routing](../../../level-7-ml-platform/06-inference-routing/)
- [Level 7 Topic 12 — kv-tiering-lmcache](../../../level-7-ml-platform/12-kv-tiering-lmcache/)
- [SGLang on GB300 NVL72: 25× perf vs H200 (Feb 2026)](https://www.programming-helper.com/tech/sglang-2026-high-performance-llm-inference-framework)
- [LMCache + NVIDIA Dynamo 1.0 (March 2026)](https://blog.lmcache.ai/en/2026/03/16/lmcache-nvidia-dynamo-1-0-a-match-made-in-inference-heaven/)
- Kiely *Inference Engineering* Ch 5 (Techniques) — the practitioner framing of every optimization above
