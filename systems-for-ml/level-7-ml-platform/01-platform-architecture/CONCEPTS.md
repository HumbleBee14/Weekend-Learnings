# 01 — Platform Architecture

## The five-box mental model

Every modern LLM platform — internal at Anthropic / Databricks / Together, or open source like vLLM Production Stack and llm-d — collapses to five boxes plus a sidecar plane.

```
                  ┌────────────────────────────────────────┐
                  │ Gateway (Envoy AI Gateway / kgateway)  │  Box 1
                  │ - Auth, token rate-limit, fallback     │
                  └─────────────┬──────────────────────────┘
                                │ ext-proc → EPP
                  ┌─────────────▼──────────────────────────┐
                  │ Inference Scheduler (KV-cache-aware,   │  Box 2
                  │ load-aware, SLA-aware)                 │
                  └──────┬──────────────────┬──────────────┘
                         │                  │
              ┌──────────▼────────┐  ┌──────▼────────────┐
              │ Prefill workers   │  │ Decode workers    │  Box 3
              │ (vLLM)            │  │ (vLLM)            │
              └──────────┬────────┘  └──────┬────────────┘
                         │   NIXL KV xfer   │
                         └──────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │ LMCache: HBM → DRAM → NVMe → remote  │       Box 4
            └──────────────────────────────────────┘

Sidecars / cluster ops (Box 5):
  Prometheus  ←  vllm:* metrics
  Grafana     ←  dashboards
  KEDA        ←  scales replicas on queue depth
  Model Reg   ←  versioned weights + eval scores; gate deploys
```

Memorise the five boxes. Every later topic in this level is implementing one of them.

## Why this shape and not another

A naive design would be: one service that loads weights, accepts requests, and replies. That works exactly until the moment any of these is true:

1. Two requests arrive at the same time and you want to batch them — that's a scheduler.
2. A request shares a prefix with the last one and you want to reuse KV — that's a KV-aware router.
3. A 70B model takes 60s to load — you need a warm pool, not scale-to-zero.
4. One tenant sends a 100K-token prompt and another sends a chat — you need fairness.
5. You want to know what p99 TTFT is — you need observability before any other lever works.

Each of these forces one of the five boxes to exist. The architecture is not a design choice; it is the residue of the constraints.

## Three reference stacks (2026)

| Stack | Gateway | Scheduler | Workers | KV tier | Autoscaler |
|---|---|---|---|---|---|
| **vLLM Production Stack** | nginx / Envoy | KV-aware router (Python or Iris/Rust) | vLLM | LMCache | KEDA on `vllm:num_requests_waiting` |
| **llm-d** (CNCF Sandbox, Mar 2026) | Gateway API Inference Extension (Envoy) | Endpoint Picker (EPP, multi-objective) | vLLM | hierarchical KV manager (LMCache compat) | Variant Autoscaler |
| **NVIDIA Dynamo** | Dynamo frontend | Smart Router (KV-aware) | TRT-LLM / vLLM / SGLang | KVBM (HBM→DRAM→NVMe→remote) | SLO Planner |

These are the same five boxes with different vendors filling each. Triton Inference Server is now legacy / maintenance — Dynamo replaces it.

References:
- vLLM Production Stack — https://github.com/vllm-project/production-stack
- llm-d architecture — https://llm-d.ai/docs/architecture
- NVIDIA Dynamo 1.0 — https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/
- Gateway API Inference Extension — https://gateway-api-inference-extension.sigs.k8s.io/

## The control plane vs data plane split

Data plane = the path your tokens travel: gateway → scheduler → worker → KV tier → back to user. Latency-critical.

Control plane = the path your *deploys* travel: registry → eval gate → rollout → scheduler config update. Throughput- and correctness-critical, not latency-critical.

Mixing them is the single most common architectural mistake. A control-plane bug should never take down the data plane. Concretely: never let the registry be in the request path. The router caches model->endpoint mappings; the registry only updates them out of band.

## What "design doc" actually means here

The deliverable for this topic is a one-page `architecture.md` you can hand someone. It must contain:

1. The five-box diagram (above), redrawn with your actual component names.
2. The data path: one sentence per hop, what gets added or removed.
3. The control path: what triggers a deploy, what gates it.
4. Two failure scenarios and how each box reacts:
   - A worker dies mid-decode.
   - A new model version fails its eval gate.
5. The cost model in one paragraph: where dollars are spent (GPU-seconds), where they're saved (KV reuse, batching).

That's it. If your design doc is longer than two pages, you are designing something else.

## What changes at 10x scale

The shape doesn't change. What changes:

- Single Prometheus → federated Prometheus (Thanos / Mimir / VictoriaMetrics). Cardinality on per-request labels eats a single-node Prom alive past about 1k QPS.
- Single registry SQLite → Postgres or a real artifact store (MLflow, Weights & Biases registry, internal "Model Hub").
- Single KEDA on one signal → multi-signal: queue depth + KV-pressure + arrival-rate prediction (Variant Autoscaler does this).
- Single-tenant → per-tenant gateway policies, per-tier dedicated worker pools.
- One region → KV cache locality stops working across regions; you shard by tenant or by prefix-hash to avoid cross-region NIXL pulls.
