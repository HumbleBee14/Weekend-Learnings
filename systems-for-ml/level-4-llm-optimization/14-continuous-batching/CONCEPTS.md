# 14 — Continuous Batching

## What it is

The fix for everything that's wrong with static batching from Level 1 Topic 03.

Static batching: collect N requests, run them as one batch, return all responses.
- **Padding waste** (mixed-length requests pad to max)
- **Head-of-line blocking** (fast users wait for slow ones)
- **Wait-vs-throughput trap** (wait longer for bigger batches → worse TTFT)

Continuous batching: the batch is a *living set*, not a snapshot. After every decode step, finished requests leave; new requests join. The batch composition changes step-by-step.

```
Static:                  Continuous:
─────────                ────────────
Step 1: [A1,B1,C1,D1]    Step 1: [A1,B1,C1,D1]
Step 2: [A2,B2,C2,D2]    Step 2: [A2,B2,C2,D2,E1]   ← E joined
Step 3: [A3,B3,C3,D3]    Step 3: [A3,C3,D3,E2,F1]   ← B finished, F joined
Step 4: [A4,B4,_,D4]     Step 4: [A4,C4,D4,E3,F2]
Step 5: ...              Step 5: ...
```

No padding waste. Fast requests don't wait for slow ones. New requests don't wait for the batch to finish.

## Why this requires paged KV

Static batching reuses contiguous memory because all requests in a batch run for the same duration. Continuous batching has requests of different ages in the same batch. The KV cache for request A might be 100 tokens long while request E just joined with 5.

Paged KV (Topic 10) solves this: each request has its own block table; blocks are independent. Batch composition changes don't disturb the per-request KV layout.

This is why paged KV came *before* continuous batching in the curriculum, and why every continuous-batching engine in 2026 uses paged KV underneath.

## The vLLM V1 scheduler

vLLM's V0 scheduler had architectural issues that became bottlenecks:
- Scheduling decisions ran on the same Python process as request handling → GIL contention
- Each step rebuilt the full batch state → CPU overhead grew with batch size
- KV cache updates and Python state tracking weren't well-separated

vLLM V1 (2025) is a rewrite:

- **Scheduler / Worker-0 separation** — different processes, communication via shared memory
- **Persistent batch** — batch state persists across steps; only diffs (which requests joined/left) are applied
- **Pinned host memory + DMA** for any data that crosses the CPU/GPU boundary
- **Disaggregated prefill and decode** scheduled in separate domains so long prefills don't block decode

The persistent batch + diff-based update is the central change. CPU overhead per step dropped 5-10× compared to V0. Throughput jumped accordingly on small models (where Python was the bottleneck).

## SGLang's overlap scheduler

SGLang took a different approach: while the GPU runs step N, the scheduler prepares step N+1 on the CPU concurrently. Prep includes: tokenization for new requests, sampling param assembly, KV cache slot allocation.

Result: the GPU never waits for the CPU. Often called the "zero-overhead scheduler."

vLLM is converging on similar techniques in 2026 (the Q2 roadmap mentions stateless scheduler + more CUDA graph capture of prep work).

## Chunked prefill — the long-context interaction

When a long prompt (>4K tokens) joins a batch, its prefill is expensive. Without chunking, the whole batch waits for that prefill to finish before any further decode steps run.

Chunked prefill: process the prompt in 4K-8K-token chunks, interleaving with decode steps for ongoing requests:

```
Step 1: prefill request E's chunk 1 (tokens 0-4095) + decode A,B,C,D
Step 2: prefill request E's chunk 2 (tokens 4096-8191) + decode A,B,C,D
Step 3: prefill request E's chunk 3 (tokens 8192-12287) + decode A,B,C,D
...
Step N: E's prefill done, E joins decode pool fully
```

vLLM has chunked prefill on by default. SGLang exposes `--chunked-prefill-size` (note: this is **batch-wide**, not per-request — common gotcha; see vllm-project/vllm#20018).

## How fairness emerges

A naive continuous batching scheduler picks "next" requests by FCFS (first come, first served). This gives reasonable fairness for similar-length requests but breaks down with mixed-length:

- Tenant A spams 10K-token prompts
- Tenant B has small chat requests
- Without fairness, B waits behind A

Real schedulers add WFQ (Weighted Fair Queueing — Level 7 Topic 07), per-tenant rate limits, priority levels.

## Pitfalls

1. **Continuous batching without paged KV.** Doesn't work — you'd need contiguous memory growing/shrinking arbitrarily.
2. **Forgetting that spec decode interacts with batching.** Spec decode produces variable token counts per step; the scheduler must handle this. Topic 17.
3. **Scheduler overhead growing with batch size.** Naive implementations rebuild state per step. vLLM V1's diff-based update is the fix.
4. **Long prefills without chunking.** Single 100K-token prefill freezes all other requests' decode. Always enable chunked prefill at long context.
5. **Confusing throughput with fairness.** Throughput at the batch level is fine; per-request fairness can still be terrible.

## What you'll do

For your `mini-vllm`:

1. Replace the static batcher (from Level 1 Topic 03) with a continuous batcher
2. The batcher loop:
   - Pop completed requests from the batch
   - Pull pending requests from the queue (up to max batch size)
   - Run one forward pass on the full batch (each request at its own KV position)
   - Sample tokens, update KV cache, mark completions
3. Run mixed-length workloads. Verify no padding waste, no head-of-line blocking.

Test:

- 1000 requests with varied lengths, concurrency 16. TTFT distribution should be tight regardless of request length.
- vs Level 1's static batching on the same workload. Throughput should jump 2-5×; TTFT for short requests in mixed batches should drop dramatically.

## References

- vLLM V1 design — https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1
- vLLM PagedAttention paper (the original continuous batching) — https://arxiv.org/abs/2309.06180
- SGLang Q2 2026 roadmap (overlap scheduler details) — https://github.com/sgl-project/sglang/issues/22949
- vLLM chunked prefill issue thread — https://github.com/vllm-project/vllm/issues/20018
- Original Orca paper (continuous batching, 2022) — https://www.usenix.org/conference/osdi22/presentation/yu
