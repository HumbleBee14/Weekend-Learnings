# Level 7 — Learning Path

The capstone level. Sixteen topics organised into five sub-arcs:

```
Architecture + control plane    (01-04)   the boxes you draw before writing code
Observability foundation        (05)      build this first; everything else is invisible without it
Routing + admission             (06-09)   the data plane's brain
Scaling + lifecycle             (10-13)   autoscaling, cold start, KV tiering, cost
Cross-cutting concerns          (14-16)   safety, reasoning workloads, RLXF closure
```

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-platform-architecture/` | 1-2h | Five-box mental model. The reference diagram every later topic implements. |
| `02-training-job-scheduler/` | 1-2h | SQLite-backed job table, PID re-attach, retry/backoff. The minimum-viable scheduler. |
| `03-evaluation-pipeline/` | 1-2h | Auto-eval on `lm-eval-harness`, regression gate. **G14 break-it list.** |
| `04-model-registry/` | 1-2h | Five-state machine, atomic promote, rollback round-trip, adapters as first-class. |
| `05-observability-otel/` | 3-4h | OTel GenAI semconv + vLLM Prometheus + DCGM + Tempo. Five-panel dashboard. |
| `06-inference-routing/` | 3-4h | KV-cache-aware router (block-hash, PrefixStore, multi-objective scorer). The biggest delta vs random. |
| `07-multi-tenant-fairness/` | 2-3h | Token rate limits + DRR / WFQ admission + cache salting + tier isolation. **G12.** |
| `08-backpressure-and-queueing/` | 2h | Little's Law as a debugging tool, bounded queue, SLO-aware admission, hedging. **G13.** |
| `09-scheduling-policies/` | 2h | FCFS / Priority+aging / SJF+aging — same workload, different tail behaviour. **G16.** |
| `10-autoscaling-keda/` | 2-3h | KEDA on `vllm:num_requests_waiting`, multi-signal triggers, graceful drain. |
| `11-cold-start-and-warmup/` | 2-3h | Run:ai Model Streamer, image pre-pull, CUDA-checkpoint, MIG vs MPS. **G15.** |
| `12-kv-tiering-lmcache/` | 3-4h | LMCache HBM→DRAM→NVMe→Redis. The four cross-replica coherence strategies. |
| `13-cost-economics/` | 2h | $/Mtok decomposed, (engine × quant × hardware) matrix, vertical-vs-horizontal. **G14.** |
| `14-safety-and-abuse/` | 1-2h | Five gateway controls, threat-model discipline, prompt injection at the infra layer. |
| `15-reasoning-aware-serving/` | 2h | What R1/o-series/Kimi K2 break: PD ratios, KV pressure, cancellation propagation. |
| `16-mini-rlxf/` | 2-3h | The full loop: trainer ↔ rollout ↔ reward ↔ buffer ↔ NCCL weight sync. Pulls every other topic. |

Total: ~30-40 hours of focused work.

## What's new in 2026 (deltas vs 2024-2025 content)

- **vLLM Production Stack + KubeRay + KEDA + Prometheus** is the canonical open-source production stack. Treated as defaults, not novelties.
- **llm-d** went CNCF Sandbox in March 2026 (Red Hat / Google / IBM / NVIDIA / AMD / HuggingFace backing). Architecture: vLLM workers + Endpoint Picker + Gateway API Inference Extension + hierarchical KV manager + Variant Autoscaler.
- **NVIDIA Dynamo** is the proprietary equivalent and replaces Triton (now legacy).
- **OpenTelemetry GenAI semconv** is the convergent observability schema; still "Development" in the spec but vendor-adopted.
- **Block-hash kv-connector** is the cross-engine standard between vLLM and LMCache.
- **NIXL** is the cross-engine KV transport library used by vLLM, Dynamo, llm-d.
- **Mooncake** joined PyTorch Ecosystem in Feb 2026 — KV disaggregation as production infrastructure.
- **vLLM Semantic Router (Iris)** in Rust (Jan 2026): +25% throughput, -1200ms TTFT vs the Python reference router.
- **KEDA on `vllm:num_requests_waiting`** is the autoscaler default; HPA-on-CPU is officially the wrong answer.
- **FinOps for AI** framework (FinOps Foundation, 2026) standardises per-tenant / per-feature / per-route attribution.
- **Reasoning models** (R1, o-series, Kimi K2.6, Claude thinking) make output-length variance, KV pressure, and cancellation propagation first-class platform problems.

## What hardware you need

- **Most of this level is CPU-only.** It is system design, orchestration, and routing logic. K8s manifests run in `kind` / `minikube` on a laptop.
- **Topic 06 / 12 / 16 want at least one GPU** to drive vLLM and observe the metrics actually move. Free Colab T4 is enough for small models.
- **Topic 11 (cold start) and 13 (cost)** benefit from a real H100/H200 hour ($2-4 on RunPod) for credible numbers.

## Project 3 closes here

Stitch:
- The Level 6 trained model (FSDP2 checkpoint) -> registered in Topic 04 -> gated by Topic 03's eval.
- The Level 5 best-of-bake-off engine -> deployed via Topic 10's KEDA + Topic 11's warmup tooling.
- Topic 06's router in front, with Topic 07's WFQ, Topic 08's backpressure, Topic 09's policy.
- Topic 05's observability stack everywhere.
- Topic 12's LMCache KV tiering for long-context.
- Topic 13's `$/Mtok` dashboards for cost.
- Topic 14's safety middleware at the gateway.
- Topic 15's cancellation propagation throughout.
- Topic 16's RLXF loop as the closing demo (architecture, not convergence).

Run the full break-it list:
- traffic skew (90% to one replica) -> WFQ on/off (G12).
- queue depth past KEDA threshold -> Little's Law overlay (G13).
- vertical vs horizontal scaling cost (G14).
- cold start during peak load (G15).
- scheduler swap FCFS vs Priority vs SJF (G16).

Ship `reports/platform.md` as a systems paper.

## After this level

- **Level 8** is the parallel local-on-device track (Apple Silicon, MLX). Same data structures, different bandwidth budgets.
- **Level 9** is the high-level compiler tour — `torch.compile` / Inductor down to PTX. Awareness, not specialisation.
- **`compiler-and-kernels/`** track is the Rust + C++ reimplementation path (Iris-style router, custom kernels). Pick this up after the curriculum if it interests you.

## What you can do at the end

> Design and ship a Kubernetes-native LLM serving platform with KV-cache-aware routing, KEDA autoscaling on `vllm:num_requests_waiting`, weighted-fair-queueing for multi-tenant isolation, OpenTelemetry GenAI semantic-convention observability, LMCache hierarchical KV offload, and per-(engine × quant × hardware) cost dashboards. Validate Little's Law against system metrics, run failure-injection (cold-start under load, scheduler swap, regression gate, traffic skew, queue threshold, cancellation propagation), and produce a systems-paper-shaped platform report.

That sentence is the deliverable. The 16 topics are how you get there honestly.
