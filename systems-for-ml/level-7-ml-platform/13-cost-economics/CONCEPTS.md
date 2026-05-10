# 13 — Cost Economics ($/Mtok and FinOps for AI)

## The single number that summarises a serving stack

Cost per million tokens — `$/Mtok` — separated for input and output. Every serving optimisation, every quantisation choice, every routing improvement, every autoscaling tweak ultimately resolves to a change in this number.

```
$/Mtok_in  = (GPU_$/hr) / (input_tokens_per_hour)
$/Mtok_out = (GPU_$/hr) / (output_tokens_per_hour)
```

Track both. They behave very differently — input throughput is set by prefill (compute-bound, batchable across requests). Output throughput is set by decode (memory-bandwidth-bound, harder to batch effectively). A model that's cheap on input can be expensive on output.

## Decomposition

Real-world `$/Mtok` is a stack:

```
$/Mtok = (GPU_$/hr + KV_tier_$/hr + observability_$/hr + warm_pool_overhead_$/hr)
        / effective_tokens_per_hour
```

- **GPU $/hr.** The dominant term. H100: ~$2-5/hr depending on vendor and commitment. B200: ~$5-10/hr. MI300X: ~$2-3/hr. Apple Silicon (Topic 8 territory): essentially 0 marginal.
- **KV tier $/hr.** Redis cluster, NVMe usage, Mooncake instance. Tens to hundreds of dollars per hour at scale; small absolute share but rising as long context becomes the norm.
- **Observability $/hr.** Prometheus / Tempo / Datadog ingest costs. At >1k QPS this becomes meaningful.
- **Warm-pool overhead $/hr.** The replicas you keep around to avoid cold start (Topic 11). At low traffic this can be the largest line item — paying for idle.

The denominator (`effective_tokens_per_hour`) is the part you actually control via:
- batch size (continuous batching saturates the GPU).
- KV reuse rate (prefix caching turns prefill seconds into cache lookups).
- precision (FP8 / NVFP4 doubles or triples effective throughput at small quality cost).
- routing efficiency (avoid prefilling shared prefixes on every replica).

## (Engine × quantisation × hardware) is the cost matrix

Project 2 (`engine-bakeoff`) gives you per-cell numbers. Project 3 turns those into a cost table:

```
                          H100         H200         B200       MI300X      L40S
                       BF16 FP8 NVFP4  BF16 FP8     BF16 NVFP4 BF16 FP8    BF16
vLLM       70B          $X   $Y   $Z    $X'  $Y'    $X'' $Z''  $X''' $Y''' $X''''
SGLang     70B          ...
TRT-LLM    70B          ...
llama.cpp  70B (CPU)    ...                                                  $K
```

The cells you don't have, you mark "unsupported" or "not measured" rather than guess. The cells you have, you cite the workload they were measured on.

## GPU utilisation is a leading indicator, not the goal

A common mistake: optimise GPU SOL (Topic 05) as the target. The actual target is `$/Mtok`. They correlate but aren't equivalent:

- A workload at 95% GPU SOL but tiny batch sizes is wasting *bandwidth* — high SOL on a kernel that's bandwidth-bound, not compute-bound.
- A workload at 60% GPU SOL with full continuous batches and FP8 might be cheaper per token than the 95%-SOL one.

Read GPU utilisation (DCGM `PIPE_TENSOR_ACTIVE`, `MEM_COPY_UTIL`) to *diagnose* `$/Mtok` regressions, not to set targets.

Healthy targets:
- Post-continuous-batching, **PIPE_TENSOR_ACTIVE >= 60%** sustained.
- **<30%** indicates bad batching, small batch sizes, or a misconfigured engine.

## Vertical vs horizontal scaling — the choice that drives **G14**

Two ways to absorb more traffic:

- **Vertical** — bigger or newer GPU. H100 → B200. Doubles throughput per replica; capex/opex per replica also goes up. Latency per request drops (FP8 / NVFP4 win on Hopper / Blackwell).
- **Horizontal** — more replicas of the same GPU. Linear capacity scaling; latency per request unchanged; cold-start cost grows with replica count.

The `$/Mtok` curves cross somewhere depending on workload:

```
$/Mtok
  │
  │  vertical (bigger GPU per replica)
  │  ────────────────╲
  │                   ╲___        crossover at high QPS where bigger
  │   horizontal       ───        GPU's better batching dominates
  │   (more replicas)
  │  ──╱
  │   ╱
  └─────────────────────────► QPS
```

For most 2026 workloads:
- At low QPS: horizontal (more cheap GPUs) wins; per-replica saturation is far away.
- At high QPS: vertical (newer GPU) wins; better batching + lower memory bandwidth pressure.
- For very-long-context workloads: vertical *plus* KV tier (Topic 12) wins; the KV memory budget is the binding constraint.

This is **G14**. Plot both curves on the same axes for your workload.

## FinOps for AI

The FinOps Foundation's "FinOps for AI" framework (2026) extends classic FinOps to ML-specific attributions:

- **Per-tenant cost.** `$/tenant/month` from `(tokens × $/Mtok) + warm_pool_share + KV_tier_share`.
- **Per-feature cost.** `$/feature/month` from spans tagged `feature=<name>`.
- **Per-route cost.** `$/route/month` from `gen_ai.request.model` × `(input + output tokens)`.
- **Per-agent-run cost.** Aggregated across all `gen_ai.client` spans within a single `gen_ai.agent` span. The right granularity for agentic workloads.
- **Forecasted cost per workflow.** Apply expected token consumption to the cost model.

The point is *attribution*. "We spent $40K on inference last month" is useless. "Tenant A's RAG workflow spent $12K, mostly on long input tokens via the GPT-class route" is actionable.

Reference: https://www.finops.org/wg/finops-for-ai-overview/

## Standard cost-related dashboards

1. **GPU utilisation vs request throughput.** Diagnoses bad batching.
2. **Tokens/sec/$ per model variant.** The bake-off comparison.
3. **Cost per tenant / per feature / per route.** Attribution. Built on OTel spans.
4. **Forecasted cost per agentic workflow run.** Predictive. For long-reasoning workloads especially (Topic 15).
5. **Warm-pool cost share.** What % of total spend is paying for idle replicas.

## Build steps

1. Plug GPU $/hr (rented or theoretical) into your Grafana dashboard alongside throughput.
2. Compute $/Mtok per (engine × quant × hardware) cell from your Project 2 numbers.
3. Compute $/Mtok per tenant from your span data.
4. **G14**: plot vertical vs horizontal scaling curves on your workload.
5. Document the FinOps approach in `reports/platform.md`.

## Pitfalls

1. **Conflating utilisation with efficiency.** High GPU SOL on a bandwidth-bound kernel is wasted. Read tensor-active and memcpy together.
2. **Single $/Mtok number for input + output.** They behave differently. Always split.
3. **Ignoring warm-pool overhead.** At low traffic this dominates. Quantify.
4. **Per-tenant cost without span-level attribution.** Approximations from QPS counts are wrong by 10-100x for mixed workloads.
5. **No cost regression alerts.** A bad deploy can spike $/Mtok 2x silently. Alert on the metric.
6. **Forecasting from small samples.** Token-distribution tails are heavy. Use p99 not mean for budget projections.

## References

- FinOps for AI — https://www.finops.org/wg/finops-for-ai-overview/
- NVIDIA H100 / B200 specs — https://www.nvidia.com/en-us/data-center/h100/
- AMD MI300X — https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- vLLM cost-aware autoscaling discussion — https://github.com/vllm-project/vllm/discussions
