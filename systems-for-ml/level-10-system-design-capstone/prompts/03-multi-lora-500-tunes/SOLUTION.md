# Prompt 03 — Worked Solution

> Open this only after attempting the prompt cold. This is one defensible design, not the only one.

## 1. Clarifying questions (the first 3 minutes)

A senior candidate asks these *before* drawing anything. They scope the design and signal you've shipped multi-tenant inference before:

1. **What's the LoRA rank distribution?** All rank 16, or mixed (8 / 16 / 32 / 64)? Mixed-rank LoRAs are the difference between "vLLM `enable_lora=True` works out of the box" and "you're punting half your fleet into a second engine." Confirm: all rank ≤ 16, single rank class.
2. **Active-set churn pattern.** Is the "50 active LoRAs" set stable across an hour, or churning every minute? (Diurnal vs. flash-crowd. Affects whether NVMe-tier lazy load is sufficient or if you need a predictive prefetcher.)
3. **Per-tenant fairness contract.** Is rate-limiting a hard ceiling (drop on exceed) or a soft weight (WFQ — serve all but starve heavy ones first)? Affects whether the router is a token bucket or a credit scheduler.
4. **Billing granularity.** Per-token, per-request, or per-GPU-second-allocated? (Per-token is the only honest answer for multi-LoRA; per-GPU-second is a lie because you're sharing the base.)
5. **Cold-tenant SLO.** "<5s cold" — does that mean first-request after model has slept 24h, or just "not in GPU memory but on local NVMe"? (The former needs S3 → NVMe warm path measured; the latter is just LoRA swap.)

**Reasonable assumptions to bake in if the interviewer waves off:**
- All LoRAs rank 16, target modules `{q_proj, k_proj, v_proj, o_proj}` — uniform shape
- Hot set (~50) drifts ~10/hour; rest are truly cold (days between hits)
- WFQ with per-tenant weights, drop only on egregious abuse (>10× weight)
- Per-token billing, separate line items for `prompt_tokens` and `completion_tokens`, tagged `lora_id` and `tenant_id`
- "Cold" = LoRA exists on object storage but not on GPU; NVMe is the warm tier

## 2. The right answer in one sentence

**One base Llama-3-8B-FP8 fleet running vLLM with multi-LoRA enabled, a consistent-hash router that pins each `lora_id` to a replica subset for cache locality, hot/cold tiering (GPU → NVMe → S3) with lazy load, and per-tenant WFQ at the gateway — *not* 500 separate deployments.**

The single sentence that separates this from the bluff answer: **it is one model with swappable adapters, not five hundred models.** Most candidates reflexively reach for "500 microservices, one per tenant" or "Kubernetes Deployment per LoRA." That answer fails on cost (500 × 1 GPU minimum = 500 GPUs for 200 QPS), fails on cold-start (every tenant pays a full container boot), and fails on utilization (450 of the 500 are idle at any moment). The correct mental model: **the base model is the substrate; LoRAs are 50MB plugins, not deployments.**

## 3. The architecture (whiteboard)

```
                          Internet
                             │
                             ▼
                  ┌────────────────────────┐
                  │   Gateway / API LB     │   ─ TLS, auth (JWT → tenant_id)
                  │   (Envoy + custom      │   ─ per-tenant token bucket
                  │    filter)             │   ─ extracts lora_id from header
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   Router (vLLM         │   ─ consistent hash on lora_id
                  │   Production Stack     │   ─ KV-cache-aware: prefers replica
                  │   + custom hash ring)  │     that already has this LoRA hot
                  │                        │   ─ falls back round-robin on miss
                  └───────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ Replica1 │    │ Replica2 │ ...│ ReplicaN │   ─ vLLM 0.7+ with
       │ Llama-3- │    │ Llama-3- │    │ Llama-3- │     enable_lora=True
       │ 8B FP8   │    │ 8B FP8   │    │ 8B FP8   │     max_loras=16
       │          │    │          │    │          │     max_lora_rank=16
       │ Hot LoRAs│    │ Hot LoRAs│    │ Hot LoRAs│   ─ GPU = H100 80GB
       │ {12,47,  │    │ {3,18,   │    │ {88,201, │     (or L40S 48GB)
       │  91,...} │    │  62,...} │    │  ...}    │
       └────┬─────┘    └────┬─────┘    └────┬─────┘
            │                │                │
            ▼                ▼                ▼
       ┌─────────────────────────────────────────┐
       │   Local NVMe — warm LoRA tier           │   ─ all 500 mirrored
       │   (per-node, 25 GB ≈ 500×50MB)          │     to every node
       └───────────────────┬─────────────────────┘
                           │ (sync on adapter publish)
                           ▼
       ┌─────────────────────────────────────────┐
       │   S3 / GCS — cold tier + source of      │   ─ versioned
       │   truth for LoRA registry               │   ─ adapter_config.json +
       │                                         │     safetensors per LoRA
       └─────────────────────────────────────────┘

       ┌─────────────────────────────────────────┐
       │   Control plane                         │
       │   ─ LoRA registry (Postgres):           │
       │       lora_id, tenant_id, version,      │
       │       s3_uri, sha256, status            │
       │   ─ Billing pipeline: OTel GenAI        │
       │       semconv events → Kafka → DWH      │
       │       tagged {tenant_id, lora_id,       │
       │       prompt_tok, completion_tok}       │
       │   ─ KEDA on vllm:num_requests_waiting   │
       │     + vllm:lora_swap_rate               │
       └─────────────────────────────────────────┘
```

### The five-box mapping
- **Gateway:** Envoy with a custom Wasm filter that decodes the JWT, extracts `tenant_id` and `lora_id`, applies per-tenant token-bucket rate limit, stamps OTel trace context. No model logic here.
- **Router:** vLLM Production Stack (the open-source K8s router) augmented with a consistent-hash table keyed on `lora_id`. For LoRA `X`, the ring picks 2–3 candidate replicas; router prefers the one that reports `X` as currently loaded. This is the cache-locality move that keeps LoRA churn low.
- **Scheduler:** vLLM's continuous batcher inside each replica. The Punica-derived multi-LoRA kernels (SGMV — segmented gather matmul vector) batch requests *across different LoRAs* in a single forward pass — this is the unlock that makes one base model serve 50 hot LoRAs at full GPU utilization.
- **Worker:** vLLM 0.7+ replicas, Llama-3-8B in FP8 (W8A8 with KV-cache FP8), `enable_lora=True`, `max_loras=16`, `max_loras_per_batch=8`. NVMe lazy load on swap-in.
- **Control plane:** Postgres LoRA registry, S3 as source-of-truth, KEDA scaling on `vllm:num_requests_waiting` and a custom `vllm:lora_swap_rate` metric (high swap rate = too few replicas for active set), OTel GenAI semconv on every request with `gen_ai.request.model = "llama3-8b"` and `gen_ai.lora.id = <lora_id>` for billing.

**The senior signal:** drawing one base-model fleet and explaining that LoRA-swap is a per-request operation (~5ms on the kernel level once weights are in GPU mem), not a deployment-level operation. The bluff answer ("500 Deployments") never recovers from that misconception.

## 4. The capacity math

```
Workload:
  200 QPS aggregate. Assume input ≈ 600 tok mean, output ≈ 200 tok mean
  (chat-shaped traffic, typical for LoRA-tuned customer assistants).

Token throughput:
  prefill  = 200 × 600 = 120K tok/s
  decode   = 200 × 200 =  40K tok/s

Llama-3-8B FP8 on H100 80GB (vLLM 0.7, single GPU, measured envelope):
  prefill   ≈ 55K tok/s
  decode    ≈ 4.5K tok/s    (at batch ≈ 96)
  concurrent decode slots ≈ 128 (KV-mem bound at 8K context)

GPU count from throughput (target 70% utilization to leave p99 headroom):
  prefill  = 120K / (55K × 0.7) = 3.1  → 4 GPUs
  decode   =  40K / (4.5K × 0.7) = 12.7 → 13 GPUs    ← decode dominates
  concurrency = (200 × ~12s lifetime) / 128 = 18.75 → 19 GPUs

Binding: 13–19 GPUs steady state. Round to 16 H100s steady,
autoscale headroom up to 24 H100s at burst.

Multi-LoRA memory overhead (the part candidates always botch):
  50 LoRAs hot × 50 MB = 2.5 GB GPU memory per replica
  Llama-3-8B FP8 weights: ~8 GB
  KV cache budget (rest of 80GB): 80 - 8 - 2.5 - 4 (CUDA graphs/runtime)
                                ≈ 65.5 GB → fits ~128 concurrent reqs at 8K ctx
  Verdict: LoRA memory is a rounding error. The S-LoRA paper proved this in 2023;
  it remains true on H100/FP8 in 2026.

Cold-load penalty (the part candidates also botch):
  Cold-tier path: S3 (50MB @ ~200 MB/s sustained) = 250ms
                  + NVMe write/cache = 50ms
                  + GPU upload via PCIe 4.0 (32 GB/s) = ~2ms
                  + vLLM LoRA registration + first-batch graph re-capture = 400-800ms
                  Total worst-case first-cold request: ~1.5s, well under 5s SLO
  Warm-tier (NVMe → GPU only): ~50ms + register = ~100ms — easily inside 500ms
  Hot (already in GPU): ~0ms swap cost — pure forward pass
```

### Cost-to-serve, blended

```
Option                                Eng-hours  $/Mtok blended  Monthly bill @ 200 QPS
──────────────────────────────────────────────────────────────────────────────────────
ONE base fleet + multi-LoRA (this)    ~120h      $0.45           $14,200
500 separate vLLM Deployments         ~60h       $11.80          $370,000+ (laughable)
Modal per-LoRA function (one fn/lora) ~40h       $4.10           $128,000
Bedrock / managed multi-LoRA          ~16h       $2.80           $87,000
```

The S-LoRA-style consolidation isn't just a clever engineering move — it's a **26× cost reduction** versus the naive design. Be ready to justify the 120h of engineering as a one-quarter payback on the first month's savings.

## 5. The hard parts — what actually breaks

### 5a. Consistent-hash routing for LoRA locality

The whole reason this design works at 200 QPS is that we don't randomly thrash LoRA weights across replicas. A request for `lora_id=47` should land on a replica that already has `47` loaded, or we pay the swap-in cost. Round-robin LB is catastrophic — at 50 hot LoRAs across 16 replicas with naive RR, every replica thrashes its working set every few seconds.

```python
# Router pseudocode (runs inside vLLM Production Stack as a custom plugin)
def route(req: Request) -> Replica:
    lora_id = req.lora_id
    # 2 candidate replicas per LoRA via rendezvous hash
    candidates = top_k_rendezvous(lora_id, all_replicas, k=2)

    # Prefer one that currently has the LoRA loaded
    loaded = [r for r in candidates if r.has_loaded(lora_id)]
    if loaded:
        return min(loaded, key=lambda r: r.queue_depth)

    # Both candidates cold — pick the one with less swap pressure
    return min(candidates, key=lambda r: r.lora_swap_rate_60s)
```

Why **rendezvous (HRW) hash** instead of plain consistent hash: replica membership changes (KEDA scale events) move ~1/N of LoRAs to new homes, not the full ring. Critical when you're autoscaling.

### 5b. S-LoRA batched kernels (SGMV)

vLLM since 0.5 ships the Punica/SGMV kernel path. It lets one batched forward pass on the base model serve requests using *different* LoRAs simultaneously — the LoRA delta is applied as a segmented matmul that gathers per-request weights. The alternative (separate forward pass per LoRA) destroys throughput.

In 2026 vLLM 0.7+, the relevant knobs:

```python
from vllm import LLM, EngineArgs
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    quantization="fp8",
    kv_cache_dtype="fp8",
    enable_lora=True,
    max_loras=16,                  # how many LoRAs can co-reside in GPU mem
    max_lora_rank=16,
    max_cpu_loras=64,              # CPU-pinned cache tier (warm-ish)
    max_num_batched_tokens=8192,
    gpu_memory_utilization=0.92,
)

# Per-request — vLLM swaps in lazily from CPU/NVMe if not already in GPU
out = llm.generate(
    prompts=[req.prompt],
    lora_request=LoRARequest(
        lora_name=f"tenant-{req.tenant_id}",
        lora_int_id=req.lora_id,        # stable int for kernel addressing
        lora_path=f"/mnt/nvme/loras/{req.lora_id}",  # NVMe path
    ),
)
```

**Reference:** Level 5 Topic 10 (`10-multi-lora-serving/`) covers the SGMV kernel in depth; the S-LoRA paper (Sheng et al., 2023) is the load-bearing citation.

### 5c. Hot/cold tiering as a three-level cache

```
TIER       MEDIUM        CAPACITY/NODE   LAT      POLICY
─────────────────────────────────────────────────────────────────────
GPU mem    HBM           16 LoRAs        ~0       LRU evict to CPU
CPU mem    DDR5          64 LoRAs        ~50ms    LRU evict to NVMe
NVMe       local SSD     ALL 500         ~250ms   stable mirror, never evict
S3         object store  ALL 500         ~500ms   source of truth, versioned
```

Every replica mirrors all 500 adapters to local NVMe at boot via a sidecar that watches the LoRA registry's Postgres notify channel. 500 × 50MB = 25 GB on disk per node — trivial. This eliminates the cold-from-S3 path for steady-state operation; S3 only matters on (a) cold boot of a new replica and (b) a newly published LoRA before sidecar sync (mitigated with a 5-minute "publishing" window before traffic is allowed).

### 5d. Per-tenant fairness and billing

```
Gateway WFQ (token-bucket, weighted by per-tenant plan):
  - Free tier:   1 req/s, burst 5
  - Pro tier:    20 req/s, burst 100
  - Enterprise:  200 req/s, burst 1000 (the heavy hitters)

Billing pipeline:
  vLLM emits OTel GenAI semconv events per request:
    gen_ai.request.model       = "llama3-8b"
    gen_ai.lora.id             = "47"
    gen_ai.tenant.id           = "acme-corp"          (custom attribute)
    gen_ai.usage.input_tokens  = 612
    gen_ai.usage.output_tokens = 184
    gen_ai.request.id          = <uuid>

  → OTel collector → Kafka topic `gen_ai.billing` → ClickHouse/DWH
  → Hourly rollup job materializes per-tenant invoice rows
```

The OTel GenAI semconv (stable as of 2025) is non-negotiable here — it's the only way to make per-tenant attribution auditable. Don't roll your own log format. Level 7 Topic 05 covers the schema.

### 5e. The "newly published LoRA" race

Customer trains a new LoRA at 14:32:00. By 14:32:01 they expect inference to work against it. Race:
1. CI uploads adapter to S3 with `version=v3`.
2. CI inserts row into `lora_registry` (Postgres) with `status=publishing`.
3. NVMe sync sidecars on every replica receive Postgres NOTIFY, pull from S3, fsync to NVMe.
4. Sidecars ACK via Postgres update; once N-of-N replicas ACKed, CI flips `status=ready`.
5. Gateway accepts requests only for `status=ready` LoRAs.

This is a 5–30s window depending on fleet size. Customers see a "publishing..." state in their console for the duration. Tell them explicitly; don't paper over it.

## 6. The break-it list

| Failure | What happens | Your mitigation |
|---|---|---|
| Single LoRA goes viral (1 customer = 80% of QPS) | One replica saturates; consistent hash sends all traffic to its 2 candidates | Detect via per-LoRA QPS, dynamically widen the candidate ring for that LoRA to k=8; sticky → spread |
| Hot set grows past 50 → 200 in an hour (campaign launch) | Swap rate spikes; p95 violates | KEDA scales on `vllm:lora_swap_rate` metric, not just queue depth; pre-warm path from event-bus signal |
| Replica OOM from KV during LoRA swap | Request fails mid-stream | vLLM swap-out budget honors KV reservation; if budget tight, request gets requeued by router (idempotent retry) |
| Adapter sha256 mismatch after S3 sync | Wrong-tenant output (catastrophic) | Gateway refuses to route to LoRAs without verified sha256 on the replica; replica advertises loaded-set with hashes |
| LoRA `v3` regresses for the tenant | Tenant complaints | Tenant-facing version pinning; default points to `latest` but tenants can pin `v2` for 30d before forced upgrade |
| Router has stale view of which replica has LoRA loaded | Sub-optimal routing, swap rate climbs | Replicas heartbeat their loaded-set every 1s via gossip → router; eventual consistency, < 2s drift |
| Noisy-neighbor: tenant submits 10K-token prompts at burst limit | Decode slots starve other tenants | Per-tenant in-flight token cap (not just QPS); enforced at gateway with a leaky-bucket on `output_tokens_in_flight` |
| FP8 quant accuracy regression on a tenant's domain | Their LoRA produces worse outputs than FP16 base | Per-LoRA quality gate at publish time vs. tenant-supplied eval set; FP8 quant verified pre-publish |
| New vLLM release breaks SGMV path | All multi-LoRA serving broken | Pin vLLM version per fleet; canary one replica for 24h before fleet upgrade |
| Postgres LoRA registry down | Can't publish new LoRAs; serving unaffected (NVMe is source) | Registry is publish-path only; serving never reads it on hot path |

## 7. What changes at 10× scale

```
At 2000 QPS and 5000 LoRAs (10× across both axes):

Substrate:
  - Single fleet no longer holds all 500 hot LoRAs; shard the LoRA space
  - Introduce a LoRA placement directory:
      lora_id → home_shard (3 shards × ~16 replicas each)
  - Router becomes two-stage:
      (1) directory lookup: lora_id → shard
      (2) consistent-hash within shard for replica
  - Per-shard scaling lets one viral LoRA's shard scale independently

Compute:
  - 130-190 GPUs by the same decode-bound math
  - Consider Llama-3.x-8B disaggregated prefill/decode (Level 5 Topic 08)
    — prefill on a small fast-pool, decode on bigger HBM-bandwidth pool
  - NVIDIA Dynamo or llm-d as the orchestrator across the disagg cluster
    (Level 5 Topic 09)

Caching:
  - LMCache (Level 7 Topic 12) for cross-replica KV reuse — RAG-style
    customers see high prefix-hit rates from system prompts
  - At 5000 LoRAs, NVMe-per-node (5000 × 50MB = 250GB) still fits but pushes
    boot time → introduce a regional LoRA cache (S3 → regional NVMe cluster)

Reliability:
  - Multi-AZ active-active; LoRA registry replicates with Postgres logical replication
  - Per-shard failure domain — losing one shard takes out 1/3 of LoRAs, not all 500

Team-shape:
  - Now justifies a dedicated multi-tenant inference SRE
  - Per-LoRA quality regression detection becomes a continuous job, not publish-time only
```

**The axis of change:** at 10× you transition from "single fleet with smart routing" to **sharded fleet with a routing directory**. The S-LoRA paper's evaluation tops out around 1000–2000 adapters per replica fleet; beyond that, sharding is the only honest move.

## 8. The 30-second summary you give the panel

> "Five hundred LoRAs on Llama-3-8B is one model with swappable adapters, not five hundred deployments. I'd run a single FP8 base-model fleet with vLLM's multi-LoRA — SGMV kernels batch across different adapters in one forward pass — sized at ~16 H100s for 200 QPS, decode-bound. Routing uses rendezvous hash on `lora_id` for cache locality, with NVMe-per-node mirroring all 500 adapters so cold-load is ~250ms not the 5s budget. WFQ at the gateway for per-tenant fairness, OTel GenAI semconv events tagged `tenant_id` and `lora_id` feed the billing pipeline. At 10× I'd shard the LoRA space with a placement directory and add disaggregated prefill/decode via Dynamo. The single biggest mistake here is reaching for one-deployment-per-tenant; that's a 26× cost regression and ignores the entire S-LoRA result."

If you deliver that cleanly in 30 seconds, you've signaled you've actually run this in production.

## What this prompt is really testing

- **Multi-LoRA as a serving primitive** (Level 5 Topic 10) — the candidate either knows SGMV or doesn't; you can't bluff it
- **Cache-locality routing** (Level 7 Topic 06) — random LB is a wrong answer here; consistent/rendezvous hash on `lora_id` is the signal
- **Three-tier storage hierarchy thinking** — GPU / NVMe / S3 with lazy promotion is the same pattern as KV-tiering (Topic 12); the candidate who's seen one sees the other
- **Per-tenant fairness + billing on a shared substrate** (Topic 07 fairness + Topic 05 OTel) — the multi-tenancy tax most candidates ignore
- **Avoiding the "one deployment per X" trap** — substrate consolidation thinking. The same instinct that says "don't build a microservice per endpoint" should say "don't build a model deployment per fine-tune"
- **Migration thinking** — knowing when the single-fleet design breaks (sharding at ~1000 hot adapters) is the seniority signal

## References

- [Level 5 Topic 10 — multi-LoRA serving](../../../level-5-production-engines/10-multi-lora-serving/)
- [Level 5 Topic 08 — disaggregated prefill/decode](../../../level-5-production-engines/08-disaggregated-prefill-decode/)
- [Level 5 Topic 09 — Dynamo + llm-d](../../../level-5-production-engines/09-dynamo-llmd/)
- [Level 7 Topic 06 — KV-cache-aware routing](../../../level-7-ml-platform/06-routing/)
- [Level 7 Topic 05 — OTel GenAI semconv](../../../level-7-ml-platform/05-otel/)
- [Level 7 Topic 07 — fairness / WFQ](../../../level-7-ml-platform/07-fairness/)
- [Level 7 Topic 10 — KEDA scaling on vLLM metrics](../../../level-7-ml-platform/10-keda/)
- [Level 7 Topic 12 — KV-tiering](../../../level-7-ml-platform/12-kv-tiering/)
- S-LoRA: Sheng et al., 2023 — the foundational result for serving thousands of LoRAs from one base
- Punica: Chen et al., 2024 — the SGMV kernel that made S-LoRA practical
- Kiely *Inference Engineering* Ch 5 §5.4 — multi-tenancy and adapter serving in production
