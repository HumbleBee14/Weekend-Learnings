# `mini-platform` — Threat Model

> One-page reference. Lives in `mini-platform/safety/threat-model.md`.

## Assets and threats

| # | Asset | Threat | Mitigation | Topic |
|---|---|---|---|---|
| 1 | Tenant A's confidential prompts | Cross-tenant prefix-cache hit | Cache salting (vLLM RFC #16016) | 07 |
| 2 | Per-tenant token budget | Abusive tenant burns through quota | Token rate-limits + AbuseCounter | 07, 14 |
| 3 | Model-emitted secrets / PII | Training-data regurgitation, model-side leak | Output PII redaction (Presidio + regex) | 14 |
| 4 | System-prompt integrity | Prompt injection via user content, tool outputs, RAG | Role tags, untrusted-content tagging, output guardrails | 14 |
| 5 | GPU availability | Resource-exhaustion abuse (long prompts, infinite outputs) | `max_tokens` cap, max prompt length, per-tenant concurrency | 07, 08 |
| 6 | Eval gate integrity | Tampered eval scores let regression deploy | Registry write-only-by-eval-runner; signed scores | 03, 04 |
| 7 | Cluster availability | Worker failure cascading to data-plane outage | KEDA `minReplicas`, graceful drain, router health-checks | 10, 11 |

## Trust boundaries

```
[ Untrusted: client request ]
        │
        ▼
[ Gateway ]  ← rate-limit, input filter, abuse counter
        │
        ▼
[ Trusted: validated prompt + tenant context ]
        │
        ▼
[ Router / Scheduler ] ← KV-aware routing, WFQ
        │
        ▼
[ Engine workers ]
        │
        ▼
[ Output stream ]
        │
        ▼
[ Output filter ]  ← Llama Guard, regex redact, PII strip
        │
        ▼
[ Untrusted: client response ]


External integrations (treat as UNTRUSTED inputs):
  - RAG vector-store contents
  - Tool-call results (HTTP fetch, search, code execution)
  - Multi-turn history echoing user content into the prompt
```

## Out of scope (for `mini-platform`)

- Network-level DDoS (assume cluster network does this).
- Insider threats inside the cluster (cluster RBAC is upstream).
- Supply-chain attacks on the model itself (training-pipeline integrity is its own topic).

## Review cadence

Reviewed quarterly or on any of:
- New tier launched (e.g., introducing self-serve API).
- New external integration (new tool, new RAG source).
- Incident retrospective revealing a new threat.
