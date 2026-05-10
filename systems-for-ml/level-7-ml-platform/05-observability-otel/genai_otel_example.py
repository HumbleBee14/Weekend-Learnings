"""
Minimal example of emitting OpenTelemetry GenAI semconv spans + metrics
from a wrapper around an LLM call. Use this as the template for your
gateway/router instrumentation.

Install:
    pip install opentelemetry-api opentelemetry-sdk \
                opentelemetry-exporter-otlp \
                opentelemetry-semantic-conventions

Run with the OTel Collector from docker-compose.yaml up.
"""

import os
import time
import random

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

OTLP = os.getenv("OTEL_ENDPOINT", "http://localhost:4317")

resource = Resource.create({
    "service.name": "mini-platform-gateway",
    "service.namespace": "ml-platform",
    "deployment.environment": "dev",
})

tp = TracerProvider(resource=resource)
tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP, insecure=True)))
trace.set_tracer_provider(tp)

reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=OTLP, insecure=True), export_interval_millis=5000
)
metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

tracer = trace.get_tracer("mini-platform")
meter = metrics.get_meter("mini-platform")

# OTel GenAI semconv metrics.
token_usage = meter.create_histogram(
    "gen_ai.client.token.usage", unit="token", description="Tokens per request"
)
op_duration = meter.create_histogram(
    "gen_ai.client.operation.duration", unit="s",
    description="End-to-end client-observed duration",
)


def call_llm(prompt: str, tenant: str, requested_model: str = "minigpt-v0.4"):
    """Wrap a single LLM call with the GenAI semconv span + metrics."""
    with tracer.start_as_current_span(
        "chat",
        attributes={
            "gen_ai.system": "vllm",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": requested_model,
            "gen_ai.request.temperature": 0.7,
            "gen_ai.request.max_tokens": 512,
            "tenant.id": tenant,
        },
    ) as span:
        t0 = time.perf_counter()
        # ... actual LLM call here ...
        time.sleep(random.uniform(0.05, 0.4))
        served_model = requested_model  # set differently if canary/fallback fired
        in_toks = len(prompt.split())
        out_toks = random.randint(32, 256)

        dt = time.perf_counter() - t0

        span.set_attribute("gen_ai.response.model", served_model)
        span.set_attribute("gen_ai.usage.input_tokens", in_toks)
        span.set_attribute("gen_ai.usage.output_tokens", out_toks)
        span.set_attribute("gen_ai.response.finish_reasons", ["stop"])

        # Common attribute set for both metric histograms.
        common = {
            "gen_ai.system": "vllm",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": requested_model,
            "gen_ai.response.model": served_model,
            "tenant.id": tenant,
        }
        token_usage.record(in_toks, {**common, "gen_ai.token.type": "input"})
        token_usage.record(out_toks, {**common, "gen_ai.token.type": "output"})
        op_duration.record(dt, common)


if __name__ == "__main__":
    for i in range(20):
        call_llm("hello world " * (i + 1), tenant=f"t{i % 3}")
    # Give the BatchSpanProcessor a moment to flush.
    time.sleep(6)
