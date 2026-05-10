# 16 — Serving Concurrency

## Files

- `CONCEPTS.md` — the five concurrency patterns real serving stacks use: sharded locks for the block manager, single-writer admission queue, cancellation propagation, stream multiplexing with bounded queues, off-event-loop tokenization

## What you do this topic

Extend your `mini-vllm` with the production concurrency patterns:

1. Sharded block manager (replace single-mutex with 8 shards)
2. `asyncio.Event` per request for cancellation
3. Per-stream bounded queues for SSE tokens
4. `asyncio.to_thread` for tokenization

This is the topic that makes `mini-vllm` survive real load, not just demos.

## Reading-driven extension

Most of the value is in reading vLLM source alongside CONCEPTS.md:

```bash
git clone https://github.com/vllm-project/vllm
cd vllm

# The four files to skim:
# 1. vllm/core/scheduler.py — single-writer pattern in action
# 2. vllm/core/block/block_manager_v2.py — sharded locks
# 3. vllm/engine/async_llm_engine.py — asyncio bridge
# 4. vllm/engine/output_processor/ — per-request output streams
```

Each pattern in CONCEPTS.md maps to specific code in those files. Trace through.

## Quick stress tests for your `mini-vllm`

After implementing the four patterns:

```python
# Test 1: sharded vs single-mutex contention
# Run 100 concurrent allocate() calls. Time them. With sharded locks, ~8× lower tail latency.

# Test 2: cancellation propagation
# Connect 100 clients, kill 30 of them halfway through. 
# Without the asyncio.Event pattern, zombie slots persist for the full max_tokens.
# With it, slots are freed within one scheduler step.

# Test 3: slow client backpressure  
# One client reading 1 token/sec, others at full speed. With per-stream bounded queues,
# only the slow client's request stalls. Without it, the whole batch backs up.

# Test 4: tokenization off the event loop
# Send 100 requests. With inline tokenization, latency for any request includes 
# tokenization time of all queued requests. With asyncio.to_thread, each request's
# tokenization runs in parallel.
```

## What you should walk away with

- A `mini-vllm` that handles real concurrent load without zombie slots, cascading slow clients, or scheduler-thread starvation
- Patterns that map directly to what vLLM does — you can read its source and recognize the constructs
- The mental model: production serving is *several concurrent loops sharing carefully-locked state*, not "one async loop"

## Where this goes

Topic 17 — speculative decoding's interaction with the scheduler. The last Level 4 topic before Project 1 closes.
