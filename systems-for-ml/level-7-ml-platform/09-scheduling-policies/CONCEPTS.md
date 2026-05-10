# 09 — Scheduling Policies

## What "scheduling" means inside a continuous-batched LLM engine

vLLM's V1 scheduler runs every step, picks which requests advance this step, and fills the batch up to the GPU's KV/compute budget. Scheduling policy is the *order* in which queued requests are considered for admission into the running batch. The policy decides who waits and who runs.

This is a different scheduler from K8s' pod scheduler. Don't confuse them. Continuous batching's scheduler runs at the **engine** level, every microbatch — millisecond cadence.

## Three policies to compare

### FCFS (first-come-first-served)

Default in most engines. Simple. Fair in the colloquial sense ("you get served in the order you arrived"). Two failure modes:

- **Head-of-line blocking.** A 100K-token prefill at the head holds the batch slot for seconds; thousands of small requests behind it all see TTFT spike.
- **No priority signal.** The free user who showed up first wins over the enterprise user who showed up second. SLO contracts can't survive that.

### Priority

Each request has a priority class. Strict-priority schedulers always pick higher-priority first. Variations:

- **Strict priority.** High starves low if high traffic is sustained.
- **Priority + aging.** Low-priority requests' effective priority increases the longer they've waited. Prevents starvation.
- **Priority + WFQ.** Each priority class has a weight; weighted round-robin within. This is what most production schedulers actually run.

In practice, priority is the gateway to tier-based SLOs (Topic 07). Free / pro / enterprise = three priority classes with WFQ weights.

### SJF-style batching (shortest-job-first)

For LLM serving, "job size" is the projected token count: prompt length + projected output length. SJF picks short jobs first; long jobs wait. Two reasons it works for LLMs:

1. **Prefill cost dominates for long prompts.** A 100K prompt can prefill in a single batch step, locking the GPU. SJF defers it until the small fast prompts have a chance to run.
2. **Mean response time is provably minimised by SJF** when service times are known (classic queueing theorem). For LLMs, service time is *partially* known (prompt length is exact; output length is bounded by `max_tokens`).

Failure mode: long jobs starve. Mitigation: aging, or a hybrid like "SJF up to a max wait time, then promote to FCFS."

## Continuous-batching-specific twists

The standard queueing-theory framing assumes one job runs at a time. Continuous batching breaks that — the engine runs *many* requests per step. The right framing is:

- Each step has a **token budget** (compute + KV memory).
- The scheduler picks a *set* of requests whose combined token cost fits the budget.
- This is a knapsack problem at every step. Engines approximate.

Practical implications:

- **Prefill chunking** matters more than scheduling policy choice. vLLM's chunked prefill splits a 100K prompt across multiple steps so it can't lock the batch. With chunked prefill on, FCFS and SJF look more similar than the textbook says.
- **Decode steps are cheap and tiny.** A scheduler that admits aggressively on decode and conservatively on prefill is implicitly doing SJF-flavoured work.

## What changes per policy on the same workload

Standard experiment: same prompt mix, same hardware, same engine. Switch the scheduling policy. Measure:

- **p99 TTFT overall.** SJF wins on mixed-length workloads. FCFS wins (slightly) on uniform workloads.
- **p99 TTFT per tenant / per priority class.** Priority+WFQ wins on tier-aware setups.
- **Throughput (tok/s).** Often within a few percent across policies — the GPU runs the same number of tokens; the policy just reshuffles whose tokens.
- **Fairness ratio.** Variance of per-tenant p99 TTFT. SJF without aging will starve the long-prompt tenant; aging fixes that.

This comparison is **G16** — required for Project 3.

## How vLLM exposes scheduling

vLLM V1 has scheduler config flags:

- `--scheduling-policy fcfs|priority` (priority class read from per-request fields).
- `--max-num-batched-tokens` — the per-step token budget.
- `--max-num-seqs` — max parallel running requests.
- `--enable-chunked-prefill` (default on in V1) — splits long prefills.
- `--preemption-mode swap|recompute` — what happens when a long-running request is preempted.

References:
- vLLM scheduler config — https://docs.vllm.ai/en/latest/serving/engine_args.html
- vLLM V1 design — https://docs.vllm.ai/en/latest/contributing/design/v1/
- SGLang scheduler — https://docs.sglang.ai/

## Per-request fields you'd add

A real scheduler needs to read per-request metadata:

```
request {
  id, prompt_tokens, max_output_tokens,
  arrival_ts,
  tenant_id, priority,         # for priority/WFQ
  estimated_service_time,      # for SJF
  deadline,                    # for EDF (rare in LLMs but exists)
}
```

Not every engine surfaces all of these; in `mini-platform` you add them at the router level and propagate as headers / per-request kwargs.

## Build steps

1. Pick two of the three policies. FCFS (you already have it) + one of (priority, SJF).
2. Implement the chosen alternative as a per-request priority field on the router queue. The router orders the queue accordingly before calling `pod.send`.
3. Workload: 10% long-prompt requests (8K-32K tokens) interleaved with 90% short-chat (200-token) requests.
4. Measure p99 TTFT for short requests under each policy. SJF should improve dramatically; priority depends on whether long requests are tagged low-priority.
5. Re-measure with vLLM's `--enable-chunked-prefill` toggled. Observe the policy difference shrink.

## Pitfalls

1. **Comparing policies without chunked prefill.** Chunked prefill is the default in vLLM V1 and changes the picture. Always include it (or its absence) in the experiment writeup.
2. **No starvation guard for SJF / strict priority.** Eventually a long prompt has to run. Add aging.
3. **Scheduling policy as the wrong lever.** If the engine is throughput-bound, swapping policy reshuffles latency but doesn't add throughput. The lever for throughput is batch size / KV efficiency.
4. **Tenant-level fairness vs request-level priority confusion.** WFQ over tenants (Topic 07) and priority over requests are different axes. Production runs both — WFQ across tenants, priority within tenant.
5. **Measuring on uniform-length workload.** Policy differences vanish on uniform workloads. The whole point of policy choice is mixed-length traffic.

## References

- vLLM engine args — https://docs.vllm.ai/en/latest/serving/engine_args.html
- vLLM V1 contributor design — https://docs.vllm.ai/en/latest/contributing/design/v1/
- SGLang scheduler — https://docs.sglang.ai/
- Sparrow / 2cR (low-latency scheduling) — https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/ousterhout
