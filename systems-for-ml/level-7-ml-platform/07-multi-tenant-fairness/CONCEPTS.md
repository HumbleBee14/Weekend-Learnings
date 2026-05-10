# 07 — Multi-Tenant Fairness

## Three layers, not one

A naive "fairness" implementation is one rate limit per tenant at the gateway. That fails the moment one tenant sends a 100K-token prompt: it stays under the QPS limit, but it monopolises GPU prefill and crushes everyone else's TTFT.

Production has three layers stacked. Each catches a class the layer above misses.

```
┌────────────────────────────────────────────────────────────────┐
│ Gateway                                                        │
│   Token-aware rate limits                                      │
│   - input_tok/min, output_tok/min, total_tok/{hour,day}        │
│   - max concurrency, max tokens per request                    │
│   Bounds worst-case GPU-seconds per tenant                     │
└────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ Scheduler / Router                                             │
│   Weighted Fair Queueing (WFQ) at admission                    │
│   - Each tenant has a weight                                   │
│   - vLLM continuous-batch admission picks next request via WFQ │
│   - Prevents one tenant's giant prompt from monopolising prefill│
└────────────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│ Hard isolation (when fairness isn't enough)                    │
│   Dedicated vLLM Deployments per tenant tier                   │
│   - free / pro / enterprise                                    │
│   - Separate KEDA policies, separate KV pools                  │
│   Real noisy-neighbor SLOs at the cost of price                │
└────────────────────────────────────────────────────────────────┘
```

The first two work for *most* multi-tenant deployments. The third is what happens when "most" isn't enough — it's how SaaS vendors deliver tier-specific SLAs.

## Why "tokens" is the right unit, not "QPS"

GPU-seconds correlate with *tokens processed*, not request count. A 100-token request and a 100K-token request have the same QPS-cost (1) and ~1000x different GPU cost. Token-aware limits — input tok/min, output tok/min, total tok/{hour,day} — are the right abstraction.

Standard set:
- `input_tokens_per_minute`
- `output_tokens_per_minute`
- `total_tokens_per_hour`
- `total_tokens_per_day`
- `max_concurrent_requests`
- `max_tokens_per_request` (single-shot abuse cap)

Envoy AI Gateway, Kong AI Gateway, and most LLM gateway products ship these primitives natively in 2026.

References:
- Envoy AI Gateway — https://aigateway.envoyproxy.io/
- Kong AI Gateway — https://docs.konghq.com/gateway/latest/ai-gateway/

## Weighted Fair Queueing (WFQ) at admission

Continuous batching's admission step normally picks the next runnable request by FCFS or arrival order. WFQ replaces that with a weighted round-robin over per-tenant queues. Each tenant `i` has weight `w_i`; over a window the tenant gets a fraction `w_i / Σw` of admission slots.

The simplest implementation is **Deficit Round Robin (DRR)**:

```
for each tenant i:
    deficit_i = 0
    quantum_i = w_i

scheduler tick:
    for each tenant i in round-robin order:
        deficit_i += quantum_i
        while head_of_queue(i) and head_size <= deficit_i:
            admit(head_of_queue(i))
            deficit_i -= head_size
```

`head_size` for LLM serving is naturally measured in *tokens* (input + projected output). Tenants get a token budget per round.

For the more nuanced **DRF (Dominant Resource Fairness)** in multi-resource settings (input tok, output tok, KV blocks), each request consumes multiple resources and the scheduler equalises each tenant's *dominant* share. Overkill for `mini-platform`; standard at large scale.

## What hard isolation actually buys

Hard isolation = different vLLM Deployments per tier. The kernel isolates them via cgroups + GPU device isolation (full GPU, MIG slice, MPS slice — Topic 11 on cold-start touches MIG/MPS). The reasons to use it:

1. **SLO contracts.** "Enterprise tier guarantees p99 < 200ms." You cannot promise this on a shared deployment because a free-tier traffic spike will violate it.
2. **Quantization tier differentiation.** Free tier on FP8 / INT4; enterprise on BF16. Different cost, different quality.
3. **Compliance.** Some tenants need data-residency or audit isolation that shared deployments can't provide.

The cost: lower utilisation. A free pool sized for peak burns 30% idle on average; a shared pool with WFQ runs at 70%+. Hard isolation is the right answer when SLO/compliance dollars exceed utilisation dollars.

## Cache salting — fairness's quiet sibling

Multi-tenant prefix caching has a cross-tenant leakage shape: tenant A's confidential system prompt becomes a *cache hit* for tenant B if their prompts happen to overlap. vLLM RFC #16016 introduces **cache salting**: include a per-tenant salt in the first block's hash input. Tenants now have disjoint prefix-cache namespaces.

Default off (because it disables cross-tenant prefix sharing, which is sometimes desirable). Turn on whenever:
- Prompts are confidential.
- Tenants are competitors.
- Compliance requires per-tenant data isolation in caches.

## Per-tenant metrics — what the dashboard must show

`tenant.id` becomes a primary dimension on every panel:

- Tokens-in / tokens-out per tenant.
- p99 TTFT per tenant (this is how you spot WFQ violations).
- Quota utilisation per tenant (used / limit).
- 429-rate per tenant (sustained 429 spikes = quota too tight or abuse signal).

Be careful with cardinality (Topic 05): if you have millions of tenants, sample or roll up to tier.

## Build steps for `mini-platform`

1. Add a `Tenant-Id` header. Reject requests without it (or assign `default`).
2. At the gateway, run a token-bucket rate limiter per tenant (tokens, not RPS).
3. At the router, replace FCFS admission with DRR over tenant queues. Default weights from a YAML config; per-request token cost = `len(prompt_tokens) + max_tokens`.
4. Workload: tenant A spams 100K-token prompts. Tenant B has chat-shaped requests. Run with FCFS, then WFQ.
5. Plot p99 TTFT for both tenants under both policies — **G12**.

## Pitfalls

1. **Per-RPS rate limits only.** A tenant inside the RPS limit can still cook the GPU. Use tokens.
2. **No max-tokens-per-request cap.** One adversarial 1M-token prompt fills the prefill window for minutes.
3. **WFQ over requests, not tokens.** A small-prompt tenant gets unfairly starved by a large-prompt tenant if the queue counts requests.
4. **Forgetting cache salting in confidential workloads.** Cross-tenant prefix hits are subtle and rarely surface until audited.
5. **Hard isolation everywhere.** Wasteful. Use it for tiers where SLO contracts demand it; share otherwise.
6. **Per-tenant cardinality bombs in metrics.** Roll up by tier in Prometheus; keep per-tenant detail in the trace backend.

## References

- Envoy AI Gateway — https://aigateway.envoyproxy.io/
- Kong AI Gateway — https://docs.konghq.com/gateway/latest/ai-gateway/
- vLLM cache salting RFC #16016 — https://github.com/vllm-project/vllm/issues/16016
- DRF paper (Ghodsi et al.) — https://people.eecs.berkeley.edu/~alig/papers/drf.pdf
