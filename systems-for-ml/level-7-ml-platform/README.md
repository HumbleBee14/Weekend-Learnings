# Level 7 — ML Platform & Production (Capstone)

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: closes **Project 3 — `mini-platform`**

## Week goal

Stitch the trained checkpoint from Level 6, the best engine from Level 5's bake-off, and a real production layer (router, autoscaler, observability, fairness, cost) into one running system. By Friday you should be able to:

- Stand up vLLM behind a KV-cache-aware router that you wrote (or configured from the vLLM Production Stack reference).
- Autoscale on `vllm:num_requests_waiting` via KEDA + Prometheus, with realistic cold-start mitigation.
- Emit OpenTelemetry GenAI semantic-convention spans and metrics; build a Grafana dashboard from them.
- Enforce per-tenant fairness with weighted fair queueing and token-aware rate limiting.
- Compute cost per million tokens per (engine × quantization × hardware) combination.
- Inject the full break-it list (cold start under load, scheduler swap, regression gate, traffic skew, queue threshold) and ship `reports/platform.md` as a systems paper.

This is the most production-shaped week. By the end you've built a miniature of what platform teams at Anthropic, Databricks, Together, Salesforce, and Red Hat actually own.

## Where this fits

- **Comes after:** Level 5 (engines), Level 6 (training).
- **Comes before:** Level 8 (local — parallel track), Level 9 (compiler tour).
- **Project this feeds:** Closes **Project 3**. Ships **G12–G17** plus `reports/platform.md`.

## 2026 reality check — the platform stack has crystallized

Three things to internalize before starting:

1. **vLLM Production Stack + KubeRay + KEDA + Prometheus is the canonical open-source production stack.** This combination appears across production ML platform architectures in 2026.
2. **llm-d (CNCF Sandbox, March 2026) is the open standard for distributed LLM serving on Kubernetes.** Architecture: vLLM workers + Inference Scheduler (Endpoint Picker) + Gateway API Inference Extension + hierarchical KV manager + Variant Autoscaler. Backed by Red Hat, Google, IBM, NVIDIA, AMD, HuggingFace.
3. **NVIDIA Dynamo is the proprietary equivalent.** Successor to Triton Inference Server (Triton is now legacy / maintenance). Components: SLO Planner, Smart Router (KV-cache-aware), KV Block Manager (KVBM, tiered HBM→DRAM→NVMe→remote), NIXL transport.

Other things that have hardened:
- **Prefill/decode disaggregation** is standard, not novel.
- **OpenTelemetry GenAI semconv** is the convergent observability schema (still "experimental" status but vendor-adopted by Datadog, Grafana, Honeycomb, New Relic).
- **KEDA scales on `vllm:num_requests_waiting`**, not CPU.
- **Token-aware rate limiting** (input tok/min, output tok/min) is the right primitive at the gateway.
- **LMCache** is the canonical KV-tier offload; default in vLLM Production Stack and llm-d.
- **NIXL** is the cross-engine KV transport library (vLLM, Dynamo, llm-d all use it).

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | platform-architecture | Design doc: training → eval → registry → serving |
| 02 | training-job-scheduler | Submit / track / fail-handle training jobs |
| 03 | evaluation-pipeline | Automated eval; regression gates |
| 04 | model-registry | Save / version / promote / roll back |
| 05 | observability-otel | OTel GenAI semconv + Prometheus + Grafana |
| 06 | inference-routing | KV-cache-aware routing; L4 vs L7; gateways |
| 07 | multi-tenant-fairness | WFQ, token-aware rate limits, per-tenant quotas |
| 08 | backpressure-and-queueing | Little's Law, validate L = λW with your metrics |
| 09 | scheduling-policies | FCFS vs priority vs SJF — measure p99 |
| 10 | autoscaling-keda | Scale on queue depth + queue latency |
| 11 | cold-start-and-warmup | Pre-warmed pools, model streamer, CRIU |
| 12 | kv-tiering-lmcache | LMCache HBM→DRAM→NVMe; long-context viability |
| 13 | cost-economics | $/Mtok per engine + quant; FinOps for AI |
| 14 | safety-and-abuse | Rate limit, prompt injection, output filtering |
| 15 | reasoning-aware-serving | Long variable outputs, cancellation, reasoning budgets |
| 16 | mini-rlxf | SFT → reward → RLHF orchestration with vLLM rollout |

### 01 — `platform-architecture`

**Output:** a one-page design doc + diagram in `mini-platform/architecture.md`. Five components, named and labeled:

```
                  ┌────────────────────────────────────────┐
                  │ Gateway (Envoy AI Gateway / kgateway)  │
                  │ - Auth, token rate-limit, fallback     │
                  └─────────────┬──────────────────────────┘
                                │ ext-proc → EPP
                  ┌─────────────▼──────────────────────────┐
                  │ Inference Scheduler (KV-cache-aware,   │
                  │ load-aware, SLA-aware)                 │
                  └──────┬──────────────────┬──────────────┘
                         │                  │
              ┌──────────▼────────┐  ┌──────▼────────────┐
              │ Prefill workers   │  │ Decode workers    │
              │ (vLLM)            │  │ (vLLM)            │
              └──────────┬────────┘  └──────┬────────────┘
                         │   NIXL KV xfer   │
                         └──────────────────┘
                                │
            ┌───────────────────▼──────────────────┐
            │ LMCache: HBM → DRAM → NVMe → remote  │
            └──────────────────────────────────────┘

Sidecars / cluster ops:
  Prometheus  ←  vllm:* metrics (num_requests_waiting, time_in_queue, ...)
  Grafana     ←  dashboards (TTFT p99, queue depth, GPU util, $/Mtok)
  KEDA        ←  scales replicas on queue depth
  Model Reg   ←  versioned weights + eval scores; gate deploys
```

This is your reference. Every later topic is implementing one box.

### 02 — `training-job-scheduler`

**Light touch.** Real platforms use SLURM, Kueue, or KubeRay job submitters. For `mini-platform`, a small Python service with a SQLite job table is enough:

- Submit: takes a config, spawns a process (or Ray job), records `RUNNING`.
- Status: read process state, surface metrics from the running job.
- Failure: detect non-zero exit, mark `FAILED`, optionally retry.

The lesson: training jobs are long, fail in many ways, and need elastic recovery. Level 6 already covered the elastic part — this topic is the orchestration around it.

### 03 — `evaluation-pipeline`

**Automated eval as a first-class platform feature.** When a training run finishes, eval runs automatically. Results land in the registry alongside the checkpoint. **Deploys are gated on eval scores.**

**Build steps.**
1. Wire `lm-eval-harness` to run on the checkpoint when `RUNNING → DONE`.
2. Save MMLU / HumanEval / GSM8K scores into the registry as metadata.
3. Define a regression rule: a new checkpoint can be promoted to `serving` only if its scores are within X% of the previous version on at least Y benchmarks.
4. Manually trigger a regression (use a worse checkpoint or skip a few training steps). Verify the gate blocks the deploy. **Part of G14's break-it list.**

### 04 — `model-registry`

**A registry is a versioned table of (checkpoint, metadata, eval scores, status).** Status flow: `staged → eval → approved → serving → retired`.

Don't overbuild. A directory of `.safetensors` plus a SQLite table with `(model_id, version, path, eval_scores, status, created_at)` is enough. The lesson is the *workflow*, not the tooling.

### 05 — `observability-otel`

**OpenTelemetry GenAI semconv** is the convergent schema. Status as of mid-2026: still marked Development/experimental, but vendor adoption is decisive (Datadog, Grafana, Honeycomb, New Relic all support it natively).

**Standard signals.**
- **Spans:** `gen_ai.client` (each model invocation), `gen_ai.agent` (agent step), `gen_ai.framework` (orchestrator).
- **Metrics:** `gen_ai.client.token.usage`, `gen_ai.client.operation.duration`.
- **Attributes:** `gen_ai.system`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.operation.name`.

**vLLM-side metrics (Prometheus).**
- `vllm:num_requests_waiting` — queue depth.
- `vllm:num_requests_running` — in-flight.
- `vllm:time_in_queue_seconds` — direct latency proxy.
- `vllm:gpu_cache_usage_perc` — KV cache pressure.
- `vllm:e2e_request_latency_seconds_bucket` — SLO histogram.

**Build steps.**
1. Run vLLM with `--otlp-traces-endpoint` and OTel GenAI semconv enabled.
2. Stand up a small OTel Collector + Prometheus + Grafana via docker-compose.
3. Build a dashboard with five panels: TTFT p99, throughput, queue depth, GPU util, KV cache fill.
4. Build a second dashboard from OTel spans: per-tenant token usage, per-model latency.

This is the foundation of every other topic this week. Without metrics, you can't see autoscaling, you can't see fairness violations, you can't compute cost.

### 06 — `inference-routing`

**KV-cache-aware routing — the algorithm.** This is the single biggest delta vs random round-robin in 2026.

1. Tokenize prompt; split into fixed-size blocks (vLLM default: 16 tokens).
2. Compute rolling prefix hashes per block. Block `i`'s key = `H(block_i_tokens + parent_hash)`. Default hash since vLLM 0.11 is **SHA-256** (collision-safe).
3. Each vLLM pod publishes the set of block hashes it currently holds via a sidecar event stream (the `kvblock.Index`).
4. The router maintains a `PrefixStore` mapping hash → set of pods.
5. On a request: walk the request's hashes from block 0 forward. At each step, intersect candidate-pod sets. Pick the pod with the longest contiguous prefix match, breaking ties by load (in-flight count, queue depth).

**Tradeoff.** Cache locality wins (TTFT drops dramatically on shared prefixes — system prompts, RAG, multi-turn). Risk: hot spots if one prefix is very popular. Production routers blend prefix score with load score (llm-d's EPP uses a multi-objective scorer).

**SGLang variant** uses RadixAttention: a radix tree of token sequences supporting O(prefix-length) match/insert/evict.

**Build steps.**
1. Two vLLM replicas. A small Python router in front (FastAPI, ext-proc, or a sidecar).
2. Implement block-level prefix hashing on the router.
3. Run a chatbot-shaped workload (shared 4KB system prompt). Measure TTFT with random routing vs prefix-aware routing.
4. Run a no-prefix-overlap workload — confirm prefix-aware ≈ random there.

Reference: vLLM Production Stack's KV-cache-aware router is open source. You can read it as a pattern.

### 07 — `multi-tenant-fairness`

**Three layers in production.**

1. **Token-aware rate limiting at the gateway.** Per tenant: input-tokens/min, output-tokens/min, total-tokens/{hour,day}, max concurrency, max tokens per request. Bound worst-case GPU-seconds per tenant. Envoy AI Gateway has token rate-limit primitives.
2. **Weighted Fair Queueing at the scheduler.** Each tenant gets a weight; vLLM's continuous-batching admission step picks the next request using DRF/WFQ rather than FCFS. Prevents one tenant's 100K-token prompt from monopolizing prefill.
3. **Hard isolation when fairness isn't enough.** Dedicated vLLM Deployments per tenant tier (free / pro / enterprise) with separate KEDA policies. Real noisy-neighbor SLOs at the cost of price.

**Build steps.**
1. Add a per-tenant `Tenant-Id` header. Track per-tenant tokens-in / tokens-out at the router.
2. Implement a simple WFQ admission policy in the router: each tenant has a weight; round-robin admit weighted by that.
3. Workload: tenant A spams 100K-token prompts. Tenant B has small chat requests. Without WFQ, B's TTFT explodes. With WFQ, B stays bounded.
4. Plot p99 TTFT for both tenants under both policies. **Part of G12.**

### 08 — `backpressure-and-queueing`

**Little's Law:** `L = λW`. Average number in the system = arrival rate × average time in the system. For LLM serving:
- `L` = `vllm:num_requests_running + num_requests_waiting`.
- `λ` = arrival rate (requests/sec at the gateway).
- `W` = average end-to-end latency.

**Why it matters.** If you measure `λ` and `W`, you can predict `L` *without observing it directly*. Conversely, observing all three lets you sanity-check your instrumentation — they should obey the law.

**Build steps.**
1. Drive the platform at a steady rate. Record λ (request/sec), W (mean latency), L (mean in-system count).
2. Compute `L = λW`. Compare to the observed `L`. They should match within ~5%.
3. Add the validation overlay to **G13** (queue depth vs latency).

**Backpressure mechanics.**
- **Queue with bound** — when full, return 429 (rate-limit) or 503 (overload).
- **Shed load proactively** — admission control rejects new requests when latency SLO is at risk.
- **Hedge** — fire a duplicate request to a second replica after a timeout; cancel the slower.

### 09 — `scheduling-policies`

**Three policies to compare.**
- **FCFS** — first come, first served. Simplest. Long requests can starve short ones.
- **Priority** — assign each request a priority (e.g., paid tier higher than free). Strict priority can starve low-priority entirely.
- **SJF-style batching** — pick the request with the shortest *expected* output first, or batch-fill with similar-length requests to minimize padding.

**Build steps.**
1. Implement two of the three in your router/admission step.
2. Run the same workload on each. Measure: p99 latency overall, p99 per-tenant, throughput.
3. **G16** — scheduling policy comparison.

### 10 — `autoscaling-keda`

**Standard 2026 pattern.**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 1
  maxReplicaCount: 8
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      query: avg(vllm:num_requests_waiting)
      threshold: "3"
```

**Why HPA/CPU autoscaling fails for LLMs.** GPU-bound workloads have constant CPU. By the time CPU-based HPA fires, the GPU queue is already deep.

**Build steps.**
1. Install KEDA in your local cluster (`kind` or `minikube` is fine).
2. Define the ScaledObject above.
3. Drive load past saturation. Watch replicas scale up.
4. Drive load to zero. Watch scale-down.
5. Measure: scale-up reaction time vs SLA breach time. **G15** (cold-vs-warm latency with scale-up window annotated).

### 11 — `cold-start-and-warmup`

**Cold-start reality.** A 70B model load is 30–120 seconds of pure I/O + CUDA context init. Scale-to-zero is a fantasy without mitigation.

**The 2026 stack of techniques (combined in production).**
- **Pre-warmed pools** — `minReplicas: 1`. Cheapest, dumbest, most common.
- **NVIDIA Run:ai Model Streamer** — concurrent streaming of safetensors directly into GPU memory, bypassing single-threaded HF loader. 40–60% faster cold starts.
- **Tiered preloading (ServerlessLLM)** — stage weights through underutilized GPU memory + local SSD. 6–8× speedup over naive load.
- **CRIU / CUDA checkpoint** — snapshot a fully-warm process (CUDA context + weights + compiled graphs in RAM) and restore. Sub-second restores achievable. Active vLLM forum experiments.
- **Warm-up requests** — after weights load, fire synthetic requests to trigger CUDA graph capture, kernel autotuning, and memory pool init before live traffic.
- **Image pre-pull / weight cache on local NVMe** — DaemonSets to pre-pull container images and cache weights so node-startup-to-ready is bounded.

**Build steps.**
1. Measure your model's cold-start time naive vs with Run:ai Model Streamer.
2. Trigger a cold start during sustained peak load (force a scale-up). Capture timeline.
3. Plot cold-vs-warm request latency with scale-up reaction window annotated. **G15.**

### 12 — `kv-tiering-lmcache`

**LMCache.** Multi-tier KV cache for vLLM:

```
GPU HBM (engine native)
   ↓ async push on eviction
CPU DRAM (pinned, hot tier)
   ↓ async push when DRAM full
Local NVMe (warm tier, large capacity)
   ↓ async push when NVMe full
Remote backend (Redis/Mooncake/InfiniStore/Ceph; persistent, slowest)
```

**Why it matters in 2026.** 128K and 1M context windows are standard. Prefilling 128K every cold request is unaffordable. Reported numbers: TTFT 11s → 1.5s for 128K system prompt on H100; up to 15× throughput on multi-turn QA / doc analysis.

**Build steps (light touch).**
1. Enable LMCache on your vLLM Deployment.
2. Run a long-doc Q&A workload — same document prefix, varied questions.
3. Measure TTFT cold (first question) vs warm (subsequent questions hitting the cached prefix).
4. Document the tier where each block lives during the run.

### 13 — `cost-economics`

**North star: $/Mtok.** Cost per million tokens, input and output separately.

**Standard dashboards.**
- GPU utilization vs request throughput (target 60–80% post-continuous-batching; <30% means bad batching).
- Tokens/sec/$ per model variant (compare H100 BF16 vs H100 FP8 vs MI300X).
- Cost per tenant / per feature / per route.
- Forecasted cost per agentic workflow run.

**Build steps.**
1. Plug GPU $/hr (rented or theoretical) into your Grafana dashboard alongside throughput.
2. Compute $/Mtok for each (engine × quant × hardware) combination from Project 2.
3. **G14** — cost vs scaling strategy (vertical = bigger GPU vs horizontal = more replicas).
4. Document FinOps approach in `reports/platform.md`. The FinOps Foundation's "FinOps for AI" framework is the standard reference in 2026.

### 14 — `safety-and-abuse`

**At the gateway, not at the model.** 2026 production runs guardrails as in-line gateway plugins (often via ext-proc, same path as the Inference Scheduler).

- **Token rate-limiting** (already in Topic 07).
- **Output filtering** — Llama Guard, NVIDIA NeMo Guardrails, PromptGuard. Run on output tokens before streaming to client.
- **Prompt injection at infra layer** — never let user content bypass system-prompt boundaries; sanitize tool-call results.
- **Abuse detection** — repeated failed attempts, suspiciously similar prompts across tenants, jailbreak signatures.

**Build steps.** Add at minimum a regex-based output filter and a per-tenant abuse counter. Document the threat model.

### 15 — `reasoning-aware-serving`

**Reasoning models break naive serving assumptions.** R1, o-series, Kimi K2, Claude thinking-mode emit very long, *highly variable* outputs. Kimi K2.6 burns 98K-token reasoning budgets per task. The Artificial Analysis benchmark = 160M reasoning tokens.

**What changes for serving infra.**
- **Output length variance breaks throughput models.** One request can hold a decode slot for minutes; another finishes in 100ms. Continuous batching handles it, but autoscaler signals should weight on `time_in_queue`, not just `num_running`.
- **Decode-heavy ratio.** Prefill is small; decode is enormous. PD disaggregation matters more — allocate more decode workers than prefill.
- **KV cache pressure.** Long outputs mean each in-flight request occupies KV blocks for the full reasoning duration. NVMe offload / LMCache becomes mandatory.
- **Reasoning budgets.** Surface knobs (`max_thinking_tokens`, `reasoning_effort`, Kimi's `preserve_thinking`, agent step caps). Gateway enforces per tenant for cost control.
- **Cancellation propagation.** Clients abandon long traces. Infra must propagate `CancelledError` from gateway → router → vLLM to free decode slots. Often missed.

**Build steps.** Add cancellation propagation to your router. Run a workload where 30% of clients disconnect mid-generation. Measure decode-slot recovery time. Document in `reports/platform.md`.

### 16 — `mini-rlxf`

**Architecture.** Trainer ↔ Rollout ↔ Reward Model ↔ Replay Buffer.

In 2026 production, **rollouts run on vLLM or SGLang**, not on the training framework. Frameworks: **verl** (most widely adopted in 2026), **OpenRLHF**, **NeMo-RL**, **TRL**.

**Build steps (light touch — show the architecture, not a full RLHF run).**
1. Sketch a tiny GRPO loop: trainer = your Level 6 setup, rollout = vLLM. Weight sync via NCCL each step.
2. Wire enough plumbing to run a few steps end-to-end on a tiny model.
3. Document the architecture, not the convergence.

## Project 3 — close out this week

```
mini-platform/
├── architecture.md              # Topic 01
├── gateway/                     # Envoy or simple FastAPI ext-proc
├── router/                      # KV-cache-aware routing logic
├── workers/                     # vLLM Deployments (configs)
├── observability/
│   ├── otel-collector.yaml
│   ├── prometheus.yaml
│   └── grafana-dashboards/
├── autoscaler/
│   └── keda-scaledobject.yaml
├── lmcache/                     # KV tiering config
├── registry/                    # SQLite + safetensors directory
├── eval/                        # lm-eval-harness wiring
├── rlxf/                        # Topic 16 sketch
└── reports/
    └── platform.md              # ← THE deliverable, written as a systems paper
```

**Required graphs (G12–G17 — finalized this week).**
- **G12** — p99 latency vs traffic skew (% to hottest replica), with WFQ on/off.
- **G13** — queue depth vs latency, with Little's Law overlay.
- **G14** — cost vs scaling strategy (vertical vs horizontal).
- **G15** — cold-vs-warm latency, scale-up reaction window annotated.
- **G16** — scheduling policy comparison (FCFS / priority / SJF) on identical workload.
- **G17** — already shipped from Level 6 (data pipeline ceiling), referenced in this report.

**`reports/platform.md`** structure (systems-paper format from outer README):
1. Problem statement.
2. System architecture (the diagram from Topic 01).
3. Methodology.
4. Key findings (numbered, quantitative claims).
5. Tradeoffs (latency vs throughput vs cost vs quality vs complexity).
6. What changes at 10× scale.

## Definition of done

- [ ] You're serving the Level 6 trained model through vLLM, fronted by a router you understand.
- [ ] KV-cache-aware routing measurably beats random on a prefix-heavy workload.
- [ ] KEDA autoscales the deployment based on `vllm:num_requests_waiting`.
- [ ] OTel GenAI semconv spans flow into your observability stack; Grafana dashboards render the canonical metrics.
- [ ] Per-tenant fairness via WFQ + token rate limits is enforced and demonstrated.
- [ ] You have all six graphs (G12–G17) with Setup/Observation/Insight captions.
- [ ] `reports/platform.md` is written as a systems paper; one of your two strongest portfolio artifacts.
- [ ] You can answer fluently: *"Walk me through how you'd serve a 70B model with 1000 QPS and a 100ms TTFT SLA on H100s."*

## Resources

- **vLLM Production Stack** — [github.com/vllm-project/production-stack](https://github.com/vllm-project/production-stack).
- **vLLM Production Stack KV-aware tutorial** — [docs.vllm.ai/projects/production-stack/en/latest/tutorials/kvaware.html](https://docs.vllm.ai/projects/production-stack/en/latest/tutorials/kvaware.html).
- **llm-d Architecture** — [llm-d.ai/docs/architecture](https://llm-d.ai/docs/architecture).
- **llm-d KV-aware routing (Red Hat)** — [developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference).
- **NVIDIA Dynamo 1.0** — [developer.nvidia.com/blog/nvidia-dynamo-1-production-ready](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/).
- **Gateway API Inference Extension** — [gateway-api-inference-extension.sigs.k8s.io](https://gateway-api-inference-extension.sigs.k8s.io/).
- **Envoy AI Gateway** — [aigateway.envoyproxy.io](https://aigateway.envoyproxy.io/).
- **OpenTelemetry GenAI semconv** — [opentelemetry.io/docs/specs/semconv/gen-ai](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
- **KEDA + vLLM autoscaling** — [docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html).
- **LMCache** — [docs.lmcache.ai/developer_guide/architecture.html](https://docs.lmcache.ai/developer_guide/architecture.html).
- **NIXL** — [github.com/ai-dynamo/nixl](https://github.com/ai-dynamo/nixl).
- **NVIDIA Run:ai Model Streamer** — [developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/).
- **FinOps for AI** — [finops.org/wg/finops-for-ai-overview](https://www.finops.org/wg/finops-for-ai-overview/).
- **verl** — [github.com/verl-project/verl](https://github.com/verl-project/verl).
- **OpenRLHF** — [github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

## Common pitfalls

1. **Building observability last.** Without metrics, every other topic is invisible. Build it first.
2. **Random routing on prefix-heavy workloads.** It works in tests, fails in production. KV-cache-aware routing is non-negotiable in 2026.
3. **HPA on CPU.** GPU-bound workloads have constant CPU. By the time CPU-HPA fires, you're 30 seconds past SLA.
4. **No cold-start mitigation.** If `minReplicas: 0`, your first request after scale-up takes 60 seconds. Always keep a warm pool unless cost forbids it.
5. **Sync-only checkpointing in the registry.** Already covered in Level 6; the registry handoff to serving has the same lesson.
6. **Skipping reasoning-aware serving.** R1/o-series/Kimi-class models break naive throughput models. Variable output length is a hard production problem.
7. **No cancellation propagation.** Half your decode slots sit on abandoned requests. Free them.
8. **No FinOps story.** "It works" + no $/Mtok number = "you cost the company $40K/month and didn't notice."

## What you'll be able to do after this week

> Design and ship a Kubernetes-native LLM serving platform with KV-cache-aware routing, KEDA autoscaling on `vllm:num_requests_waiting`, weighted-fair-queueing for multi-tenant isolation, OpenTelemetry GenAI semantic-convention observability, LMCache hierarchical KV offload, and per-(engine×quant×hardware) cost dashboards. Validate Little's Law against system metrics, run failure-injection (cold-start under load, scheduler swap, regression gate, traffic skew, queue threshold, cancellation propagation), and produce a systems-paper-shaped platform report.
