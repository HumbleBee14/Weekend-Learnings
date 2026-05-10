# 08 — Backpressure and Queueing

## Little's Law as a debugging tool

```
L = λ · W
```

In a stable system: average number in the system = arrival rate × average time in the system. For LLM serving:

- `L` = `vllm:num_requests_running + vllm:num_requests_waiting` (averaged).
- `λ` = arrival rate at the gateway (req/s).
- `W` = average end-to-end latency (s) — `vllm:e2e_request_latency_seconds`.

If you measure all three from your own metrics, they should obey the law within a few percent. They almost always don't on the first try, and *that* is the lesson — the discrepancy reveals an instrumentation bug.

Common failures:
- `λ` measured at the wrong layer (router instead of gateway: ignores rejected/queued-out-of-band requests).
- `W` excludes time spent queued at the gateway (only measures vLLM-side).
- `L` undercounts because preemptions move requests in and out of `running` rapidly.

Fix instrumentation until L = λW holds. Now you have a calibrated system. From there, the law lets you *predict* one quantity from the other two — for example, you can estimate p99 latency under a target QPS by measuring `L` at fixed concurrency.

## What backpressure actually does

Backpressure is the umbrella term for "the system pushes back when it cannot keep up." Three distinct mechanisms:

### Bounded queue + 429 / 503

Cheapest, dumbest, most common. Set `max_queue_depth`. When full, return HTTP 429 (rate-limit) or 503 (overload). Clients with retry budgets back off; the system's tail latency stays bounded.

```
if queue_depth >= max_queue_depth:
    return 429
else:
    queue.put(request)
```

The trap: pick `max_queue_depth` carelessly and you either reject too eagerly (waste capacity) or too late (queue tail latency dominates). Tune from observed p99 — if your TTFT SLO is `S` seconds and the system's saturation throughput is `T` rps, the queue should hold about `S × T - in_flight` items.

### Admission control with SLO awareness

Queue based on whether the SLO can still be met. A request that arrives when the queue would push its end-to-end latency past `S` is rejected up front, freeing the system to serve faster requests on time. This is what NVIDIA Dynamo's SLO Planner does, and what Sema4's "early shed" pattern does.

```
predicted_W = current_queue_depth / current_throughput
if predicted_W + service_time > S:
    return 429
```

The hard part is `service_time` — for LLMs it depends on prompt length and projected output length. You don't know `output_length` until the model decides. So in practice you bound it (`max_tokens` cap) and use that as the upper bound of `service_time`.

### Hedging

Fire a duplicate request to a second replica after a short timeout (`p95_latency × 1.5`). Cancel whichever is slower. Reduces tail latency without raising mean throughput much. Used selectively — hedging a hot prefix to a *cold* replica re-prefills the prefix, blowing efficiency. Smart hedge: pick a replica that already shares some of the prefix (Topic 06's PrefixStore is the input).

## Where the queue lives

```
Client ──► Gateway ──► Router ──► Worker (vLLM scheduler)
              │           │            │
            queue?      queue?       queue (the one that always exists)
```

The vLLM scheduler queue exists by design (continuous batching). The other two are choices.

- **Gateway queue.** Rare. Usually the gateway only does TLS + auth + rate-limit and immediately forwards. Putting a queue here adds a hop's worth of latency.
- **Router queue.** Useful. The router can apply WFQ (Topic 07), prefix-aware sticky routing, hedging — all of which are more easily expressed when there's a queue to inspect.
- **Worker queue.** Always exists (`vllm:num_requests_waiting`). Cannot be eliminated without losing continuous batching.

The signal you autoscale on (Topic 10) is the worker queue. The signal you backpressure on is whichever queue is your control point — usually the router queue if you have one, otherwise the worker queue read via Prometheus.

## Concrete numbers (rough but useful)

Order-of-magnitude defaults to start from on a single H100 serving an 8B model with reasonable continuous batching:

- saturation throughput ~80-200 req/s
- saturation TTFT (when queue is non-empty) follows `queue_depth × per_step_time / batch_size`
- a queue depth of ~3 per replica is "warming up"
- a queue depth past ~10 per replica is "tail latency is exploding"

Numbers vary by model and hardware; don't trust these for production tuning. Measure your own.

## Hedging in detail (when it's worth the cost)

Cost of hedging: extra prefill on the hedge replica. For prefix-heavy workloads, that prefill is *cheap* if the hedge replica also holds the prefix; *expensive* otherwise. The right hedge policy reads the PrefixStore:

```
if p95_inflight_time exceeded:
    hedge_to = best_other_replica_with_prefix(request)
    if hedge_to.matched_prefix > 50% of total:
        send_hedge(hedge_to)
```

In production at large scale (Anthropic / OpenAI / Google), hedging is a tier-conditional feature — enterprise tier hedges, free tier doesn't.

## Build steps

1. Drive the platform at a steady, sub-saturation rate. Record `λ`, `W`, `L`. Compute `L_pred = λW`. Compare. They should match within ~5%.
2. Increase `λ` past saturation. Plot `L` vs `W` — the queue depth vs latency curve. **G13.**
3. Add a bounded queue at the router. Confirm 429 emission once depth hits the cap.
4. Add a hedge: after p95 latency without first token, dispatch a hedge to the best-prefix-match other replica. Measure TTFT p99 with and without.

## Pitfalls

1. **Ignoring rejected requests in λ.** If your gateway rate-limits before the queue, those requests are *not* arrivals to the system. Be deliberate about which boundary you're measuring.
2. **Mixing prefill-bound and decode-bound regimes.** L = λW holds in either, but the *throughput model* changes. Decode-bound systems on reasoning-heavy traffic have very different `W` distributions (Topic 15).
3. **Cancelled requests still consuming GPU-seconds.** Without cancellation propagation (Topic 15), your `L` undercounts because the metric drops the request but the GPU keeps decoding.
4. **Hedging too eagerly.** Sending a hedge after p50 latency doubles GPU spend with little tail benefit. Hedge after p95+.
5. **Queueing forever.** A request that has been queued past its SLO is dead-on-arrival. Either timeout-and-reject early or use SLO-aware admission.

## References

- Little's Law (a careful exposition) — https://en.wikipedia.org/wiki/Little%27s_law
- The Tail at Scale (Dean & Barroso) — https://research.google/pubs/pub40801/
- NVIDIA Dynamo SLO Planner — https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/
- Sema4 / load-shedding patterns — https://sre.google/sre-book/handling-overload/
