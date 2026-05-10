# 14 — Continuous Batching

## Files

- `CONCEPTS.md` — what continuous batching solves vs static (padding waste, head-of-line blocking), why it requires paged KV, vLLM V1 scheduler architecture, SGLang's overlap scheduler, chunked prefill

## What you do this topic

Replace the static batcher in your `mini-vllm` with a continuous batcher.

## Reading-driven extension

Most of the value is in CONCEPTS.md and in extending your existing `mini-vllm`:

```python
class ContinuousBatcher:
    def __init__(self, model, kv_cache, max_batch_size: int):
        self.model = model
        self.kv_cache = kv_cache  # paged from Topic 10
        self.max_batch_size = max_batch_size
        self.active_requests: dict[int, RequestState] = {}
        self.pending_queue: deque[Request] = deque()

    def step(self):
        # Pop completed requests
        completed = [rid for rid, st in self.active_requests.items() if st.is_done()]
        for rid in completed:
            self.kv_cache.free_request(rid)
            del self.active_requests[rid]

        # Pull new requests up to max batch size
        while len(self.active_requests) < self.max_batch_size and self.pending_queue:
            req = self.pending_queue.popleft()
            self.kv_cache.allocate_request(req.id, n_tokens=len(req.prompt_tokens))
            self.active_requests[req.id] = RequestState(req)

        # Run one forward pass on the full batch
        # (each request at its own position in its own KV slot)
        token_outputs = self.model.forward_batched(
            requests=list(self.active_requests.values()),
            kv_cache=self.kv_cache,
        )

        # Sample, append KV, update state
        for req_state, new_token in zip(self.active_requests.values(), token_outputs):
            req_state.append_token(new_token)
            # KV cache append happens inside model.forward_batched
```

## Quick benchmark plan

Compare your continuous batcher to Level 1's static batcher on the same workload:

```python
# Workload: 100 requests, varied lengths (50-500 input tokens, 50-500 output tokens)
# Concurrency: 16

# Static batching from Level 1 Topic 03
# vs
# Continuous batching with paged KV (this topic)
```

Expected outcomes:

- Throughput: continuous wins 2-5× (no padding waste)
- TTFT for short requests in mixed batches: dramatically better with continuous
- Memory usage: similar in absolute terms, much higher utilization with continuous

## Try

- **Inject a 5000-token prompt into a batch of 50-token prompts.** Static would block everyone; continuous interleaves with chunked prefill (if you implement it).
- **Profile both batchers** with Level 3 tools. The shape of the GPU utilization curve is dramatically different.
- **Compare to vLLM V1**. Same workload. vLLM should win on absolute numbers (it's hand-tuned); your `mini-vllm` should match the *shape* of the throughput curve.
- **Measure scheduler overhead.** With Python's GIL and naive scheduler, CPU overhead per step grows with batch size. vLLM V1's diff-based update is the production fix; you can implement a simplified version.

## Where this goes

After this topic, `mini-vllm` is recognizably a real serving engine: paged KV + prefix sharing + continuous batching + (optional) speculative decoding. The remaining Level 4 topics cover edge cases and refinements:

- Topic 15 — structured output (JSON schema, grammar masking)
- Topic 16 — serving concurrency primitives (locks, queues, cancellation)
- Topic 17 — spec decode systems integration

Project 1 closes here: `mini-vllm` with all of the above, benchmarked against the Level 1 baseline (static batching) and against production vLLM.
