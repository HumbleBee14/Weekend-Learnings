# 07 — Multi-Tenant Fairness

## Files

- `CONCEPTS.md` — three-layer fairness (gateway token-rate-limits, router WFQ, hard isolation), why tokens not QPS, DRR vs DRF, cache salting, per-tenant metrics.
- `wfq_admission.py` — gateway token bucket + router-side DRR admission. Run as a demo.
- `quotas.yaml` — declarative per-tenant config (free / pro / enterprise example).

## Quickstart

```bash
python wfq_admission.py
```

## Expected output (shape)

```
[admitted] a-0 tenant_a cost=200
[admitted] a-1 tenant_a cost=200
[admitted] a-2 tenant_a cost=200
[admitted] a-3 tenant_a cost=200
[admitted] b-big tenant_b cost=4000
[admitted] a-4 tenant_a cost=200
...
```

The point: `tenant_b`'s 4000-token request does *not* monopolise admission. It gets its proportional share by weight and the small-prompt `tenant_a` requests interleave correctly.

## Try

- **FCFS comparison.** Replace DRR with a single FIFO queue. Re-run. The 4000-token request blocks all `tenant_a` requests behind it. This is the picture you draw under FCFS.
- **Tighten the gateway.** Drop `tenant_b`'s `output_tok_per_min` to 1000. Watch the gateway 429 most of `tenant_b`'s requests before they reach the router.
- **Wire to Topic 06.** Call `admit.submit(tenant, cost)` inside `router.pick`'s wrapper before forwarding upstream. Drive both tenants through `bench.py` and capture per-tenant TTFT for **G12**.
- **Hard isolation.** Add an `isolation: dedicated` branch in the router that routes enterprise-tier requests to a separate pod set. Free/pro share the main pool with WFQ; enterprise gets dedicated workers.

## G12 measurement plan

- Two tenants, weights 1:1 (or 1:4 — your call).
- Tenant A: chat-shaped (200-token prompts).
- Tenant B: long-prompt-shaped (8K-token prompts).
- Drive 5 minutes each at FCFS and at WFQ.
- Capture per-tenant p99 TTFT.
- The expected picture: under FCFS, A's p99 follows B's worst-case prefill time. Under WFQ, A's p99 stays bounded; B's p99 rises slightly.

## Where this goes

- Topic 08: backpressure decisions are tenant-aware — shed load by tier.
- Topic 09: WFQ is a scheduling policy variant; comparison with FCFS / priority / SJF feeds **G16**.
- Topic 14: token rate-limits are also the abuse mitigation primitive.
