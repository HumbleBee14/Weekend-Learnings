# 05 — Observability (OpenTelemetry GenAI semconv)

## Files

- `CONCEPTS.md` — the convergent OTel GenAI schema, vLLM Prometheus metrics, DCGM, the five-panel dashboard, why HPA-on-CPU fails for LLMs.
- `docker-compose.yaml` — OTel Collector + Prometheus + Tempo + Grafana, single command to bring it up.
- `otel-collector.yaml` — receivers, redaction processor, tail sampling, exporters.
- `prometheus.yaml` — scrapes vLLM `/metrics`, DCGM-exporter, Collector self-metrics.
- `tempo.yaml` — local trace storage.
- `grafana-datasources.yaml` — auto-provisioned Prometheus + Tempo data sources.
- `grafana-dashboards/llm-five-panel.json` — the canonical dashboard: TTFT, throughput, queue depth, GPU SOL, KV fill.
- `genai_otel_example.py` — template instrumentation for a gateway/router span.

## Quickstart

```bash
docker compose up -d
# Grafana:    http://localhost:3000   (anonymous Admin)
# Prometheus: http://localhost:9090
# Tempo:      http://localhost:3200

# In a separate shell, run vLLM with OTLP traces enabled:
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --otlp-traces-endpoint http://localhost:4317

# Or run the example without vLLM:
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp \
            opentelemetry-semantic-conventions
python genai_otel_example.py
```

## Expected output

- Prometheus: `vllm:num_requests_waiting`, `vllm:time_to_first_token_seconds_bucket`, `mini_platform_gen_ai_client_token_usage_*` all visible.
- Tempo: traces named `chat` with `gen_ai.system=vllm`, `gen_ai.usage.input_tokens`, `tenant.id`.
- Grafana: open the "LLM five-panel" dashboard; panels render once vLLM serves a few requests.

## Try

- **Cardinality stress.** Add `request_id` as a metric attribute. Watch Prometheus memory climb. Remove it.
- **Redaction.** Add `gen_ai.prompt` to the example span's attributes. Confirm the Collector's `attributes/redact` processor strips it before export.
- **Sampling.** Change `tail_sampling.probabilistic.sampling_percentage` and watch Tempo's trace volume change.
- **Dashboard from spans.** Add a Grafana panel that queries Tempo for `tenant.id=t0` to compare token usage across tenants.

## Where this goes

- Topic 06: router emits `gen_ai.client` spans with `gen_ai.response.model` set after pick.
- Topic 08: validate Little's Law using `vllm:num_requests_running + waiting` vs `λ` and `W`.
- Topic 10: KEDA reads `vllm:num_requests_waiting` from this Prometheus.
- Topic 13: cost dashboards built on the same token-usage histogram, joined to `(model, quant, hardware)`.
