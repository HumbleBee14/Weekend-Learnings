# 09 — Dynamo and llm-d

The 2026 production frontier. These aren't engines; they're **orchestration layers** above engines like vLLM, SGLang, and TRT-LLM. They do the routing, autoscaling, KV transport, and disaggregation control that an engine alone can't.

## NVIDIA Dynamo

> "An inference operating system for AI factories."

Production 1.0 in 2026. NVIDIA's open-source orchestration framework for LLM serving at fleet scale. Customer-facing as part of NIM.

What Dynamo is, in components:

```
┌──────────────────────────────────────────────────────────────────┐
│ Dynamo control plane                                              │
│  - placement (which engine, which GPU, prefill vs decode pool)    │
│  - KV-cache-aware router (routes to worker with prefix in cache)  │
│  - autoscaler (per-pool, queue-depth-driven)                      │
│  - health, metrics, traces (OpenTelemetry GenAI)                  │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Worker pools                                                      │
│  ┌─────────────────────┐   ┌──────────────────────────┐           │
│  │  Prefill workers     │   │  Decode workers           │           │
│  │  TRT-LLM / vLLM      │   │  TRT-LLM / vLLM           │           │
│  │  large batch, SM-bound│  │  many slots, HBM-bound    │           │
│  └─────────────────────┘   └──────────────────────────┘           │
│                                                                   │
│            └──── NIXL / NCCL P2P / RDMA ────────┘                 │
│            (KV transfer between pools)                            │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ NVIDIA Inference Microservices (NIM) — the customer wrapper       │
│  Containerized; ships pre-tuned engines per (model × GPU × prec) │
└──────────────────────────────────────────────────────────────────┘
```

Key Dynamo features:

- **Engine-agnostic.** Backs TRT-LLM, vLLM, SGLang. You're not locked into TRT-LLM.
- **NIXL** for KV transport. Open-source library; works over NVLink, RDMA, TCP.
- **KV-cache-aware routing.** Aggregates prefix-cache state from all workers; routes to maximize hit rate.
- **Disaggregated by default.** Prefill and decode pools are first-class.
- **Multi-model.** One Dynamo deployment serves many models; placement is dynamic.

## llm-d

CNCF Sandbox (March 2026). The open-source equivalent on Kubernetes. Backers: Red Hat, IBM, Google, NVIDIA.

llm-d's architecture mirrors Dynamo's at the conceptual level, but built on Kubernetes primitives + Envoy AI Gateway:

```
┌──────────────────────────────────────────────────────────────────┐
│ Envoy AI Gateway (the L7 router, OpenAI-compatible)              │
│  - extProc plugin holds the KV-cache-aware routing logic         │
│  - exposes Prometheus metrics + OTel GenAI traces                │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Inference Gateway API (Kubernetes CRDs)                           │
│  InferencePool / InferenceModel / TrafficSplit                    │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Worker pods (vLLM by default; pluggable)                          │
│  prefill pool + decode pool, each with KV connector to LMCache    │
└──────────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ LMCache cluster                                                   │
│  shared KV blocks, block-hash kv-connector standard               │
└──────────────────────────────────────────────────────────────────┘
```

Key llm-d features:

- **Kubernetes-native.** Pods, services, HPA, the whole thing.
- **Envoy AI Gateway** for L7 routing. Same gateway used by other AI control planes.
- **Inference Gateway API** — the standard CRDs for InferencePool/InferenceModel; this is where the K8s ecosystem is converging.
- **LMCache by default** for KV transport (instead of NIXL). Block-hash kv-connector standard.
- **Inference Scheduler** — KV-aware scheduling extension for kube-scheduler.

## Side-by-side

```
                         Dynamo                   llm-d
                         ──────                   ─────
Stewardship              NVIDIA                   CNCF (Red Hat, IBM, Google)
Control plane            Custom                   Kubernetes-native (CRDs)
Default KV transport     NIXL                     LMCache
Engine support           TRT-LLM, vLLM, SGLang    vLLM (primary), pluggable
Customer surface         NIM containers           Helm charts on K8s
Standardization          De facto (NVIDIA)        CNCF Sandbox + InferenceGW API
Observability            OTel GenAI               OTel GenAI + standard K8s metrics
Disaggregation           First-class              First-class
Multi-model              Dynamic placement        InferencePool CRDs
```

In practice, the two converge at the architecture level (KV-aware routing, disagg, multi-model). The differentiator is the deployment substrate. NVIDIA-shop on bare metal? Dynamo. Kubernetes-native? llm-d.

## Why this matters for the curriculum

You won't run either at full scale this week. But the **patterns** they implement are exactly what Level 7's `mini-platform` will mimic at smaller scale:

```
Dynamo / llm-d component       mini-platform equivalent (Level 7)
──────────────────────────     ──────────────────────────────────
KV-cache-aware router          Topic 06 — inference routing (sticky sessions)
Per-pool autoscaler            Topic 10 — queue-depth-driven autoscaling
Disaggregated worker pools     Topic 06 — separate prefill/decode endpoints
LMCache shared KV              Topic 08 — shared cache layer
OpenTelemetry GenAI traces     Topic 05 — observability stack
Multi-tenant fairness          Topic 07 — WFQ across tenants
Cost dashboard                 Topic 12 — $/Mtok dashboard
```

You build the toy version of each. By the time you've shipped Level 7, you can read a Dynamo or llm-d architecture doc and know what every component is for.

## What to do this topic

1. Read the Dynamo overview docs and the llm-d overview docs (links below). 30 min each.
2. Look at the vLLM Production Stack quickstart — it's the simplest "deploy vLLM + LMCache + Envoy AI Gateway on K8s" path. You can run it on a single-node Kind cluster locally.
3. Sketch a one-page architecture diagram of either Dynamo or llm-d. Mark which components correspond to which Level 7 topics. This is what `architecture_compare.py` prints.

## Pitfalls

1. **Treating Dynamo as "TRT-LLM-only."** It serves vLLM and SGLang too.
2. **Treating llm-d as "vLLM-only."** vLLM is the primary engine, but the CRDs and Inference Gateway API are engine-agnostic by design.
3. **Confusing NIM with Dynamo.** NIM is the containerized distribution; Dynamo is the orchestration framework. NIM uses Dynamo internally.
4. **Skipping the InferenceGW API.** It's the K8s-side standard the rest of the ecosystem is converging on. Knowing it pays off in Level 7.
5. **Building on top of either as a learner.** Both are appropriate for production fleets, not for week-5 exploration. Use them as architecture references; build the toy in Level 7.

## References

- NVIDIA Dynamo home — https://docs.nvidia.com/dynamo/latest/
- NVIDIA Dynamo source — https://github.com/ai-dynamo/dynamo
- NIXL (KV transport) — https://github.com/ai-dynamo/nixl
- NIM home — https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/
- llm-d home — https://llm-d.ai/
- llm-d source — https://github.com/llm-d/llm-d
- llm-d in CNCF Sandbox — https://www.cncf.io/projects/llm-d/
- Inference Gateway API — https://gateway-api-inference-extension.sigs.k8s.io/
- vLLM Production Stack — https://docs.vllm.ai/projects/production-stack/en/latest/
- LMCache architecture — https://docs.lmcache.ai/developer_guide/architecture.html
- Envoy AI Gateway — https://aigateway.envoyproxy.io/
