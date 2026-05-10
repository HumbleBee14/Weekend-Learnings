# 05 — Observability (OpenTelemetry GenAI semconv)

## The convergent schema

Through 2024 every vendor invented their own attribute names: `prompt_tokens`, `completion_tokens`, `model_name`, `latency_ms`. Through 2025 OpenTelemetry's **GenAI semantic conventions** went from draft to widely-adopted. Through 2026 it is *the* schema for LLM telemetry. Status is still "Development / experimental" in the spec, but every major vendor (Datadog, Grafana, Honeycomb, New Relic, Splunk) ships native dashboards on it.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/

## What you emit

**Spans (traces).** Three span kinds.

| Span name | When |
|---|---|
| `gen_ai.client` | one model invocation (chat/completion request to an LLM) |
| `gen_ai.agent` | one agent step (often wraps multiple `gen_ai.client` calls) |
| `gen_ai.framework` | one orchestrator turn (LangGraph node, etc.) |

**Metrics (Prometheus / OTLP).**

| Metric | Type | Meaning |
|---|---|---|
| `gen_ai.client.token.usage` | histogram | tokens per request, with `gen_ai.token.type=input|output` |
| `gen_ai.client.operation.duration` | histogram | end-to-end client-observed duration |

**Standard attributes** (subset that matters most):

| Attribute | Example |
|---|---|
| `gen_ai.system` | `vllm`, `openai`, `anthropic` |
| `gen_ai.operation.name` | `chat`, `text_completion`, `embeddings` |
| `gen_ai.request.model` | `meta-llama/Llama-3-8B-Instruct` |
| `gen_ai.response.model` | the actual model that served (post-canary, post-fallback) |
| `gen_ai.usage.input_tokens` | 128 |
| `gen_ai.usage.output_tokens` | 312 |
| `gen_ai.response.finish_reasons` | `[stop]`, `[length]`, `[content_filter]` |
| `gen_ai.request.temperature` | 0.7 |
| `gen_ai.request.max_tokens` | 1024 |

The split between `request.model` (what you asked for) and `response.model` (what served) is essential when canary deploys, fallback chains, and A/B tests are in play. Always set both.

## vLLM-side metrics (Prometheus)

These are the engine-internal counters you scrape. Names as of vLLM v0.11+:

| Metric | What it tells you |
|---|---|
| `vllm:num_requests_running` | in-flight requests |
| `vllm:num_requests_waiting` | queue depth — **the autoscaler signal** |
| `vllm:time_in_queue_seconds` | direct latency-floor proxy |
| `vllm:gpu_cache_usage_perc` | KV cache pressure |
| `vllm:cpu_cache_usage_perc` | LMCache DRAM tier pressure |
| `vllm:e2e_request_latency_seconds_bucket` | SLO histogram |
| `vllm:time_to_first_token_seconds_bucket` | TTFT histogram |
| `vllm:time_per_output_token_seconds_bucket` | ITL histogram |
| `vllm:prompt_tokens_total` | input throughput counter |
| `vllm:generation_tokens_total` | output throughput counter |
| `vllm:num_preemptions_total` | scheduler thrash signal |
| `vllm:request_prefill_time_seconds_bucket` | prefill latency |
| `vllm:request_decode_time_seconds_bucket` | decode latency |

`num_requests_waiting` is the single most important number on this list. KEDA scales on it (Topic 10), backpressure decisions read it (Topic 08), and SLO panels divide on prefill vs decode using its companions.

## Why HPA on CPU fails for LLMs

GPU-bound workloads have nearly constant CPU utilisation. By the time a CPU-based HPA fires (cooldown + threshold + smoothing), the GPU queue is already deep and TTFT has already broken SLO. Always scale on `num_requests_waiting` (or `time_in_queue` for SLA-aware policies).

## The five-panel dashboard (start here)

```
┌──────────────────────┬──────────────────────┐
│  TTFT p50 / p95 / p99│  Throughput tok/s    │
│  (vllm:ttft_seconds) │  (gen_tokens / dur)  │
├──────────────────────┼──────────────────────┤
│  Queue depth         │  GPU SOL %           │
│  (num_requests_      │  (DCGM gpu_utilization│
│   waiting)           │   + sm_active_ratio) │
├──────────────────────┴──────────────────────┤
│  KV cache fill %     (gpu_cache_usage_perc) │
│  + LMCache tier hits (Topic 12)             │
└─────────────────────────────────────────────┘
```

Five panels. Build this *first*. Without it, every other topic in this level is invisible.

## A second dashboard from spans (per-tenant, per-model)

Built on top of OTel `gen_ai.client` spans, queried from your trace backend (Tempo / Jaeger / Honeycomb / Datadog APM):

- Tokens-in / tokens-out per `tenant.id` (split by `gen_ai.token.type`).
- p99 latency per `gen_ai.request.model`.
- Finish-reason distribution (`stop` vs `length` vs `content_filter`) — content-filter spikes are abuse signal.
- Per-route `$/Mtok` (joins to Topic 13's cost table).

## GPU metrics — DCGM is the source

NVIDIA DCGM-Exporter exposes GPU utilisation, SM activity, memory utilisation, power, temperature as Prometheus metrics. Two distinctions matter:

- `DCGM_FI_DEV_GPU_UTIL` — wall-clock fraction the GPU is doing *something*. Misleadingly high.
- `DCGM_FI_PROF_SM_ACTIVE` — fraction of SMs actually executing work. The honest signal.
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` — fraction the tensor cores are busy. The number that maps to GPU $/hr efficiency.

"GPU SOL" (speed-of-light) is internal NVIDIA jargon for utilisation against theoretical peak. In practice your dashboard panel reads `PIPE_TENSOR_ACTIVE` for tensor-bound ops, plus `MEM_COPY_UTIL` for the bandwidth-bound parts of decode.

Reference: https://github.com/NVIDIA/dcgm-exporter

## OTel Collector — where everything lands

```
   vllm engine  ──► OTLP traces  ──┐
   gateway      ──► OTLP traces  ──┤
   router       ──► OTLP traces  ──┼─►  OTel Collector  ──┬─►  Tempo / Jaeger
                                   │                      │
   vllm engine  ──► /metrics    ───┤                      └─►  Prometheus
   DCGM-exporter──► /metrics    ───┘                          │
                                                              ▼
                                                          Grafana
```

The Collector is the right place to:
- redact PII from span attributes,
- sample (head-based or tail-based),
- enrich (add `cluster`, `region`, `model_version` from a config),
- dual-export (one Prometheus, one cost-tracking sink, one long-term archive).

## Build sequence

1. `docker-compose` for OTel Collector + Prometheus + Tempo (or Jaeger) + Grafana.
2. Run vLLM with `--otlp-traces-endpoint http://otel-collector:4317`.
3. Provision a dashboard with the five panels above, plus a per-tenant token-usage panel.
4. Fire load through the gateway. Confirm spans appear in Tempo, metrics in Prometheus, panels render.
5. Validate Little's Law (Topic 08) using `λ = rate(prompt_tokens_total)`, `W = avg(e2e_latency)`, `L = avg(num_requests_running + num_requests_waiting)`.

## Pitfalls

1. **Building observability last.** Without metrics, every other topic this week is hand-waving. Build the five-panel dashboard first.
2. **Cardinality bombs.** Putting `request_id` or full prompts in span attributes (or worse, in metric labels). Prometheus dies past ~1M unique label combinations.
3. **No `response.model` attribute.** Every canary/A-B test now requires guessing which version served the request.
4. **Ignoring DCGM tensor-active.** "GPU at 95%" via `GPU_UTIL` while tensor cores idle = a misconfigured engine. Always check `PIPE_TENSOR_ACTIVE`.
5. **Single Prometheus past 1k QPS.** Federate (Thanos / Mimir / VictoriaMetrics) before you have to.
6. **Logging tokens.** PII-laden by default. Redact at the Collector, not at the producer.

## References

- OpenTelemetry GenAI semconv — https://opentelemetry.io/docs/specs/semconv/gen-ai/
- vLLM Prometheus metrics — https://docs.vllm.ai/en/latest/serving/metrics.html
- DCGM-Exporter — https://github.com/NVIDIA/dcgm-exporter
- OTel Collector — https://opentelemetry.io/docs/collector/
- Tempo (Grafana traces) — https://grafana.com/docs/tempo/latest/
