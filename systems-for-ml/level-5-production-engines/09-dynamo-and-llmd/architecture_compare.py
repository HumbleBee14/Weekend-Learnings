"""
09 - Print a side-by-side ASCII map of Dynamo vs llm-d, plus the mapping
from each platform's components to the Level 7 mini-platform topics they
correspond to.

This is a reading aid, not a runtime. Use it as a checklist while reading
the Dynamo and llm-d architecture docs.
"""

DIAGRAM = r"""
╔═══════════════════════════════════════════════════════════════════════╗
║          NVIDIA Dynamo                       llm-d                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   Steward    NVIDIA                          CNCF Sandbox             ║
║   Substrate  bare metal / K8s                Kubernetes-native        ║
║   Gateway    Triton / Dynamo router          Envoy AI Gateway         ║
║   KV xport   NIXL                            LMCache                  ║
║   Engines    TRT-LLM, vLLM, SGLang           vLLM (pluggable)         ║
║                                                                       ║
║                  ┌──────────────────────────────────────┐             ║
║                  │  L7 Router  (KV-cache-aware)         │             ║
║                  │  Dynamo: built-in                    │             ║
║                  │  llm-d: Envoy AI Gateway + extProc   │             ║
║                  └──────────────────────────────────────┘             ║
║                                  │                                    ║
║                                  ▼                                    ║
║              ┌────────────────────────────────────────┐               ║
║              │  Inference Pool API                    │               ║
║              │  Dynamo: native                        │               ║
║              │  llm-d: InferencePool CRDs (K8s)       │               ║
║              └────────────────────────────────────────┘               ║
║                  │                       │                            ║
║                  ▼                       ▼                            ║
║         ┌────────────────┐   ┌────────────────────┐                   ║
║         │ Prefill workers │   │ Decode workers     │                   ║
║         │  TRT-LLM/vLLM   │   │  TRT-LLM/vLLM      │                   ║
║         └────────────────┘   └────────────────────┘                   ║
║                  │                       ▲                            ║
║                  └─── KV transfer ───────┘                            ║
║                  Dynamo: NIXL          llm-d: LMCache                 ║
║                                                                       ║
║              ┌────────────────────────────────────────┐               ║
║              │ Observability                          │               ║
║              │ Both: OpenTelemetry GenAI + Prometheus │               ║
║              └────────────────────────────────────────┘               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

LEVEL7_MAP = """
Component in Dynamo / llm-d           ↔   mini-platform topic (Level 7)
─────────────────────────────────────     ─────────────────────────────
KV-cache-aware L7 router              ↔   06 — inference routing
Per-pool autoscaler                   ↔   10 — autoscaling
Disaggregated worker pools            ↔   06 — separate endpoints
Shared KV cache (LMCache / Mooncake)  ↔   08 — backpressure & queueing
OpenTelemetry GenAI traces            ↔   05 — observability
Multi-tenant fairness                 ↔   07 — multi-tenant fairness
Cost / $/Mtok dashboard               ↔   12 — cost economics
Cold-start prewarming                 ↔   11 — cold start & warmup
Regression-gate before deploy         ↔   03/04 — eval pipeline + registry

Read the Dynamo and llm-d architecture docs while keeping this map next
to you. By the time you finish Level 7, every component on the left has
a working toy in your repo.
"""


def main() -> None:
    print(DIAGRAM)
    print(LEVEL7_MAP)


if __name__ == "__main__":
    main()
