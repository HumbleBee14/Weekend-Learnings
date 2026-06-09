# Prompt 01 — Worked Solution

> The most "everything we taught" prompt in the set. At cloud-provider scale, there's no clever substrate trick — you have to actually know the full Level 7 stack and assemble it.

## 1. Clarifying questions (the first 3 minutes)

A senior candidate asks these *before* drawing anything. They scope the design and signal you've sized real platforms before:

1. **Model catalog growth.** Adding 1 new model / week or 1 / month? (Affects deploy automation depth — eval-gate pipeline becomes critical at higher cadence.)
2. **Per-class latency SLO.** Is the 800ms TTFT target the same for 7B and 70B-class, or different? If same, *disaggregated prefill/decode (Dynamo)* becomes mandatory for the 70B path. If different (e.g., 2s for 70B), classical vLLM is fine.
3. **Multi-region topology.** Active-active across 5 regions, or active-passive (US-East primary)? (Affects KV cache replication strategy, billing reconciliation, failure-domain math.)
4. **Billing granularity.** Per-token, per-request, or compute-time? If per-token (the customer-friendly answer), the OTel pipeline carries every `input_tokens` / `output_tokens` span downstream to invoicing. If compute-time, simpler but customers hate it.
5. **Isolation requirement.** Can two customers share a GPU (logical multi-tenancy via vLLM scheduler) or does each request need MIG-level hardware isolation? (Affects fleet count by ~2-3×.)
6. **Catalog composition.** All open-source models or any closed-weight third-party? (Affects compliance, key management.)

**Assumptions to bake in if waved off:** 30 models in catalog, +1/week growth; 800ms TTFT for ≤13B, 2s for 70B; active-active in US-East/US-West/EU, active-passive in APAC/India; per-token billing; logical multi-tenancy is acceptable; all open-source weights.

## 2. The right answer in one sentence

**Per-region Kubernetes deployments running vLLM Production Stack for ≤13B models + NVIDIA Dynamo 1.0 with disaggregated prefill/decode for 70B-class + Envoy AI Gateway as the L7 entry + KV-cache-aware routing via dynamo-router + per-tenant weighted-fair-queueing in the scheduler + KEDA autoscaling on `vllm:num_requests_waiting` + OpenTelemetry GenAI semconv spans flowing through Kafka into a billing/usage pipeline, with LMCache as the cross-region KV tier for hot prefixes.**

This is the only prompt in the set that justifies the full Level 7 stack. At managed-cloud scale, K8s and owned infra win on every axis — cost, isolation, latency, compliance. The senior signal is *naming the specific 2026 production tools* (Dynamo 1.0, vLLM Production Stack, LMCache, Envoy AI Gateway) rather than vague "Kubernetes and microservices."

## 3. The architecture (whiteboard)

```
                            Customer ─── HTTPS ──► Anycast DNS
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │   Per-region edge    │
                                              │   (US-E, US-W, EU,   │
                                              │    APAC, India)      │
                                              └──────────┬───────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │  Envoy AI Gateway    │  Box 1 — Gateway
                                              │  ─ TLS / mTLS        │  ─ OpenAI-compat
                                              │  ─ API-key auth      │     shim
                                              │  ─ per-tenant rate-  │  ─ OTel injection
                                              │    limit (req + tok) │  ─ region pinning
                                              │  ─ prompt-injection  │     for compliance
                                              │    pre-filter (L7.14)│
                                              └──────────┬───────────┘
                                                         │
                                                         ▼
                              ┌──────────────────────────────────────────────────┐
                              │  dynamo-router  (Dynamo 1.0)                     │  Box 2 — Router
                              │  ─ tracks KV cache placement across worker pool  │  vLLM Production
                              │  ─ KV-aware: routes to replica that already has  │  Stack's KV-aware
                              │    the prefix (5-12× cost reduction at 60-85%    │  router descends
                              │    hit rate on chat-shaped workloads)            │  from this
                              │  ─ per-tenant WFQ across requests                │
                              │  ─ model-aware: dispatches to right worker pool  │
                              └──────────┬───────────────────────────────────────┘
                                         │
                ┌────────────────────────┼─────────────────────────────────┐
                │                        │                                 │
                ▼                        ▼                                 ▼
       ┌──────────────┐         ┌─────────────────┐               ┌──────────────────┐
       │ Worker pool  │         │  Worker pool    │               │ Disagg pool      │
       │ ≤13B vLLM    │         │  Embedding /    │               │ for 70B class    │
       │ (per model)  │         │  reranker /     │               │ (Dynamo)         │
       │              │         │  vision (Triton │               │                  │
       │ ─ FP8/NVFP4  │         │  IS, L5.15)     │               │ ┌──────────────┐ │
       │ ─ continuous │         │                 │               │ │ Prefill pool │ │
       │   batching   │         │                 │               │ │ (B200, small)│ │
       │ ─ EAGLE-3    │         │                 │               │ └──────┬───────┘ │
       │   spec decode│         │                 │               │        │ NIXL    │
       └──────────────┘         └─────────────────┘               │        ▼         │
                                                                  │ ┌──────────────┐ │
                                                                  │ │ Decode pool  │ │
                                                                  │ │ (MI300X, big)│ │
                                                                  │ └──────────────┘ │
                                                                  └──────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ LMCache cross-      │  Box 4 — KV tier
                              │ region KV tier      │  (Topic 12)
                              │ ─ HBM → DRAM → NVMe │
                              │ ─ hot-prefix repl   │
                              │   across regions    │
                              │ ─ via Dynamo KV     │
                              │   offloading        │
                              └─────────────────────┘
                                         │
                                         ▼ (all components emit OTel)
                              ┌──────────────────────────────────────┐
                              │  Control plane                       │  Box 5
                              │  ─ Argo CD: Helm chart per model     │
                              │  ─ Model registry (L7.04): MLflow    │
                              │  ─ Eval gates (L7.03): lm-eval-      │
                              │    harness + per-model task suites,  │
                              │    block deploy on >2% regression    │
                              │  ─ Prometheus + Grafana + Tempo      │
                              │  ─ KEDA: scale on                    │
                              │    vllm:num_requests_waiting         │
                              │  ─ Billing pipeline:                 │
                              │    OTel → Kafka → BigQuery →         │
                              │    aggregator → invoice              │
                              └──────────────────────────────────────┘
```

### Five-box mapping
- **Gateway:** Envoy AI Gateway (open-source, built by Bloomberg + Tetrate in the CNCF Envoy community; proposed as a Linux Foundation / Agentic AI Foundation project, GB vote pending as of mid-2026 — it is *not* a CNCF "graduating" project; the core Envoy proxy is the graduated one). Owns auth, TLS, per-tenant rate-limiting (both request-rate and token-rate), OpenAI API compatibility shim, OTel injection, and the prompt-injection pre-filter from L7.14.
- **Router:** dynamo-router. Tracks per-worker KV state via KV events; routes new prefixes to the most-cache-resident replica. This is the single highest-leverage component in the stack — KV-aware routing yields **5-12× cost reduction** on chat-shaped workloads at typical 60-85% hit rates.
- **Scheduler:** Inside each vLLM worker, the engine's continuous batcher does fine-grained scheduling. Across workers, per-tenant WFQ in the router gives fairness (L7.07).
- **Worker:** Two pools per region. (a) Standard vLLM pool, one deployment per ≤13B model. (b) Disagg pool via Dynamo for 70B-class: small prefill pool on B200 (compute-bound), large decode pool on MI300X (bandwidth-bound; HBM3e wins for decode), connected via NIXL.
- **Control plane:** Argo CD-driven; model deploys are Helm chart commits. Eval gates block bad deploys. Billing pipeline siphons every OTel span tagged with `tenant_id`, `model_id`, `input_tokens`, `output_tokens` into Kafka → BigQuery for invoicing.

## 4. Capacity math (per region, illustrative)

Assume 30 models in the catalog with a Pareto distribution: top 5 models = 80% of traffic. Per region peak:

```
Aggregate per-region peak: 2,000 QPS    (top-5 models dominate)
Mean input / output:       800 / 300 tokens
Top 5 hot models:          Llama-3-70B (40%), Llama-3-8B (20%),
                           Qwen-2.5-32B (10%), Mistral-7B (5%), Llama-3-405B (5%)
Long tail (25 models):     ~20% of traffic, mostly ≤13B

PER MODEL — example for Llama-3-70B (40% of traffic = 800 QPS):

  input_tok/s   = 800 × 800   = 640,000 tok/s
  output_tok/s  = 800 × 300   = 240,000 tok/s

  Engine perf (Dynamo + disagg, your bake-off confirms):
    prefill on B200:  72,000 tok/s/GPU   (compute-bound)
    decode  on MI300X: 1,500 tok/s/GPU   (bandwidth-bound)
    concurrent slots: 64/GPU

  Prefill GPUs needed     = 640K / (72K × 0.70)   = 12.7 → 13 B200
  Decode  GPUs needed     = 240K / (1.5K × 0.70)  = 228  → 228 MI300X
  Concurrency GPUs        = (800 × 16s) / 64      = 200  → 200

  Binding = decode @ 228 GPUs
   × 1.3 (p99 headroom)   = 297
   + 2   (N+2 redundancy) = 299
   × 1.2 (warm pool)      = 359 MI300X for Llama-3-70B decode pool

  Plus 13 × 1.3 × 1.2 = 21 B200 for prefill pool.

PER REGION TOTAL (all 30 models summed, with the disagg/standard mix):
  ~720 MI300X decode capacity  (mostly Llama-3-70B + 405B)
  ~  60 B200 prefill capacity
  ~ 180 H100 for ≤13B standard pool
  ~ Total ~960 GPU equivalent per region at peak.

FIVE REGIONS:  ~4,800 GPU equivalent at peak across the fleet.
WITH COMMITTED-USE 50% / on-demand 30% / spot 20% mix:
  blended $/hr ≈ $1.80 (committed) + $3.50 (on-demand) + $0.90 (spot, batch)
```

### Cost story per region

```
Per region per day (Llama-3-70B example, 40% of traffic, 800 QPS peak / 200 trough):
  tokens/day  = ~75 billion
  $/Mtok blended = ~$0.85 with FP8 + spec decode + 70% prefix-cache hit
                    on chat-shaped workloads
  daily compute cost   ≈ $64K
  monthly compute cost ≈ $1.9M for this one model in this one region

Catalog-wide, all regions: ~$45-60M/month compute. KV tier + observability + bandwidth: +~$3M/month.
```

The interviewer doesn't need exact numbers, but they want to see: prefill / decode separation, the headroom multiplier chain (1.3 × 1.1 × 1.2 ≈ 1.7× over raw), and an order-of-magnitude cost answer.

## 5. The hard parts that distinguish a 4 from a 3

### 5.1 Per-tenant billing pipeline

This is half the system for a managed offering, and it's where bluffers fail. Spec:

```
vLLM emits OTel GenAI semconv spans:
  span.attributes = {
    gen_ai.system: "vllm",
    gen_ai.request.model: "llama-3-70b",
    gen_ai.usage.input_tokens: 847,
    gen_ai.usage.output_tokens: 312,
    tenant.id: "acme-corp-prod",
    request.id: "req_01HZX...",
    duration_ms: 1842,
  }

Envoy AI Gateway injects:
  tenant.id (from API key lookup)
  region (from region-pinning policy)

Pipeline:
  OTel collector → Kafka topic `gen-ai-usage` (1 partition per region) →
  Flink/Spark Streaming consumer →
  BigQuery `usage_events_v1` (raw) + `usage_5min_rollups` (aggregated) →
  Invoicing service (monthly) + Customer dashboard (real-time)

  Late-arriving spans (the network-failure case): 7-day reconciliation window;
  finalized invoice cuts at +30 days.
```

The senior signal: **acknowledging that this pipeline has its own SLA** (you can't bill customers if Kafka is down; you have to choose between drop-and-credit-later vs. block-traffic-on-billing-down).

### 5.2 KV-aware routing — the single biggest cost lever

A naive round-robin router hashes by `request_id`. Cache hit rate ≈ random ≈ low. dynamo-router (descended from vLLM Production Stack's `kv-aware-router`) routes by prompt prefix hash, sending requests with shared prefixes to the same replica. On chat workloads (multi-turn with growing history), this yields **60-85% cache hit rate**, and a cache hit saves the entire prefill cost.

At our 800 QPS for Llama-3-70B with mean 800-token input, 70% prefix-cache hit = ~70% of prefill cost eliminated = ~$45K/day saved on this one model in this one region = **~$13M/year saved across the fleet**. Naming this specific optimization with this specific dollar figure is the strongest possible signal in the interview.

### 5.3 Disaggregated prefill/decode for the 70B path

A 70B model on a single H100 has unbalanced compute: prefill is compute-bound (saturates the tensor cores), decode is bandwidth-bound (waits on HBM). Co-locating them on the same GPU wastes one or the other at any moment.

Dynamo's disagg pattern: tiny pool of prefill-optimized GPUs (B200, max FLOPS) handles prefill, ships the KV state via NIXL (low-latency RDMA) to a much larger pool of decode-optimized GPUs (MI300X, max HBM bandwidth). Result: ~30% lower total $/Mtok at this scale, plus the prefill and decode pools scale *independently* with workload shape.

Crucially this only pays off above some QPS threshold (the NIXL transfer adds 5-15ms latency). For ≤13B models or low QPS, classical vLLM is the right answer. The interviewer wants you to *name the threshold* and explain *why* you use disagg only for the 70B-class hot path.

### 5.4 Multi-region KV tier via LMCache

LMCache (March 2026 Dynamo 1.0 integration) gives you persistent KV storage that survives replica restarts and replicates hot prefixes across regions. The use case: system prompts (long, shared across customers) live in cross-region KV; per-conversation KV stays region-local.

Without this, a customer hitting the EU region after starting in US-East pays full prefill cost again. With it, the EU region's first response on the existing conversation is a cache hit, not a re-prefill.

### 5.5 Catalog automation (the "+1 model per week" tax)

```
Adding a new model is a PR to the catalog repo:

  models/qwen-3-omni/
    helm-values.yaml         # engine version, GPU type, replicas, env vars
    eval-config.yaml         # task subset for lm-eval-harness
    routing-policy.yaml      # which workload mix this model targets

  CI on PR:
    1. Helm chart lint
    2. Stage deploy in canary cluster
    3. Run eval-gate (45 min)
    4. Compare to baseline; block if regression > 2% on any task
    5. Smoke test 100 requests; verify schema + latency
    6. Manual approval → Argo CD promotes to production

  Rollout: 5% traffic for 24h → 50% for 24h → 100%.
```

This is the difference between "we can launch a model in a sprint" and "we can launch a model in an afternoon."

## 6. Break-it list

| Failure | What happens | Mitigation |
|---|---|---|
| Region outage (e.g. US-East AZ-1) | 20% capacity loss; routing must fail over | Anycast DNS + per-region active-active; N+1 region capacity headroom; traffic shifts within 2 min |
| dynamo-router crash | All traffic dies (single point of failure) | Run 3+ router replicas behind Envoy; sticky-session affinity tolerated to lose for cache-hit degradation, not for availability |
| Billing pipeline (Kafka) lag/down | Invoices delayed; some spans lost | Local span buffering on collector for 24h; reconcile within 7-day window; SEV-2 page if >1h lag |
| Bad model promote | Customers see regression or 5xx | Eval-gate blocks; canary deploys at 5%/24h catch it; auto-rollback on >10× error-rate spike |
| Cold start during 5× traffic burst | New replicas take 70s (L7.11) | Pre-warmed pool sized for 20% margin; KEDA fires on `num_requests_waiting > N` 30s before SLA breach; image pre-pulled via DaemonSet; Run:ai model streamer cuts cold-start in half |
| Single tenant DDoS (one customer 100× their normal rate) | Noisy-neighbor: their workload starves others | Per-tenant token-rate limits at Envoy; WFQ in router enforces share; hard cap on concurrent requests per tenant |
| Disagg KV transfer (NIXL) saturation | 70B path TTFT spikes | NIXL has bandwidth headroom monitoring; if usage > 80%, scale decode pool (more receivers); if persistent, add a second prefill→decode mesh per region |
| New CVE in vLLM | Have to patch the whole fleet | Engine version pinned per model deploy; canary upgrade path; rolling update by model (not all at once) takes ~6 hours fleet-wide |
| Compliance audit (SOC2) | Need request lineage | Every span has tenant_id + request_id + model + region + timestamp; queryable for 13 months in BigQuery |
| Sudden EU GDPR enforcement on a customer | Need to ensure their data never leaves EU | Region-pinning policy at Envoy enforces routing; KV tier respects region tags; auditable via OTel spans |

## 7. What changes at 10× scale (20K QPS aggregate per region)

This is the seniority signal. At 10× the design above, several axes change:

**Hardware diversification.** Today: H100 / B200 / MI300X. At 10×: add custom silicon for the long tail — Trainium for ≤7B cheap-tier models, TPU v6 for non-NVIDIA shop customers, Gaudi3 for bursty workloads. The orchestration layer becomes hardware-aware (each model has a `compatible_hardware` list, scheduler chooses cheapest available).

**Cross-region KV federation.** LMCache federated tier: hot prefixes (system prompts, shared instructions) replicated to all 5 regions. Per-conversation KV stays region-local. The cross-region replication is async, low priority — eventual consistency is fine for cache.

**Speculative decoding by default.** EAGLE-3 in vLLM V1 was experimental at our scale; at 10× it's mandatory. Be precise about the gain: the textbook 2× is a low-concurrency figure, but a high-QPS fleet runs compute-bound at large batch where dense-model spec decode delivers closer to ~1.3–1.5× (long-context multi-turn pushes it back toward 2× because decode stays bandwidth-bound). Call it ~25–35% fewer decode GPUs fleet-wide — still a very large absolute saving at this scale, just not a clean halving.

**Reasoning-aware serving as a separate product** (L7.15). o1-class models break the standard SLO model — long decode, KV pressure, cancellation propagation. Different pool, different pricing tier ($X/Mtok output × 10 for reasoning tokens), different scheduler.

**A dedicated platform team per region.** Org-shape change, not tech. At this scale you can't centrally operate; each region gets its own platform team + on-call rotation. The model catalog automation has to be self-service for product teams.

**Custom kernels.** Cross-fleet, a 2% inference speedup is $30M/year. Now you have an in-house compiler team (sibling `compiler-and-kernels/` track) contributing Triton kernels back to vLLM/SGLang for the workload patterns you see most.

**Multi-fabric networking.** InfiniBand within DC, Ultra Ethernet between DCs in the same metro, dedicated fiber across regions. The networking topology becomes a design discipline of its own.

## 8. The 30-second summary

> "I'd build five active-active per-region Kubernetes deployments. Standard ≤13B models on vLLM Production Stack with KV-aware routing — that one optimization is worth $13M/year at our hit rates. The 70B-class path is Dynamo with disaggregated prefill/decode — B200 for prefill, MI300X for decode, NIXL between. Envoy AI Gateway as the entry, per-tenant WFQ in the router, KEDA on `num_requests_waiting`. The billing pipeline is OTel GenAI semconv → Kafka → BigQuery, with a 7-day reconciliation window. LMCache federated tier for cross-region KV on hot prefixes. At 10× scale we add custom silicon for the long tail, speculative decoding by default, and a dedicated platform team per region. Per-region fleet is roughly 960 GPU equivalent at peak; catalog-wide compute runs ~$45-60M/month."

If you can deliver that in 30 seconds, you're hired.

## What this prompt is really testing

- **Integration depth across the full Level 7 stack.** No clever substrate trick rescues you — you have to know all of it.
- **Naming specific 2026 tools** — Dynamo 1.0, vLLM Production Stack, LMCache, Envoy AI Gateway, KEDA, OTel GenAI semconv. Vague "Kubernetes" answers fail.
- **Billing pipeline awareness** — half the system for a managed offering. Bluffers omit it entirely.
- **KV-aware routing as the highest-leverage optimization** — naming this in dollars-per-year terms is the strongest signal.
- **Disagg threshold judgment** — knowing *when* to disaggregate (70B-class hot path, not the long tail).
- **10× scale answer that names axes of change** — not just "more GPUs."

## References

- [Topic 06 — inference-routing](../../../level-7-ml-platform/06-inference-routing/) — KV-aware routing details
- [Topic 08 — backpressure-and-queueing](../../../level-7-ml-platform/08-backpressure-and-queueing/) — Little's Law for the queue math
- [Topic 12 — kv-tiering-lmcache](../../../level-7-ml-platform/12-kv-tiering-lmcache/) — LMCache cross-region details
- [Topic 13 — cost-economics](../../../level-7-ml-platform/13-cost-economics/) + [CAPACITY-PLANNING.md](../../../level-7-ml-platform/13-cost-economics/CAPACITY-PLANNING.md)
- [Level 5 Topic 08 — disaggregated-inference](../../../level-5-production-engines/08-disaggregated-inference/)
- [Level 5 Topic 09 — dynamo-and-llmd](../../../level-5-production-engines/09-dynamo-and-llmd/)
- [NVIDIA Dynamo 1.0 docs](https://docs.nvidia.com/dynamo/)
- [LMCache + Dynamo 1.0 integration (March 2026)](https://blog.lmcache.ai/en/2026/03/16/lmcache-nvidia-dynamo-1-0-a-match-made-in-inference-heaven/)
- [vLLM Production Stack on GitHub](https://github.com/vllm-project/production-stack)
- [llm-d (CNCF Sandbox 2026)](https://github.com/llm-d/llm-d)
- Kiely *Inference Engineering* Ch 7 (Production) for the practitioner framing
