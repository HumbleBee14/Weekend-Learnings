# 16 — Serving Concurrency

## Why this topic exists

Production LLM serving stacks aren't "one async batcher loop." They have several concurrent processes/threads sharing state, and the locking strategy decides whether the system scales or melts.

This topic is the concurrency *patterns* — sharded locks, single-writer queues, cancellation propagation, stream multiplexing — that vLLM and SGLang actually use under their continuous-batching layer.

## The layout of a real serving stack

```
                ┌─────────────────────────────┐
HTTP requests ─→│  HTTP handlers (asyncio)    │  N tokio workers / asyncio tasks
                └────────────┬────────────────┘
                             │ submit
                             ▼
                ┌─────────────────────────────┐
                │  Pending request queue      │  thread-safe / lock-free
                └────────────┬────────────────┘
                             │ pop
                             ▼
                ┌─────────────────────────────┐
                │  Scheduler thread (one)     │  the single writer of batch state
                │  - allocates KV blocks      │
                │  - admits to active batch   │
                │  - runs forward pass        │
                └────────────┬────────────────┘
                             │ produces tokens
                             ▼
                ┌─────────────────────────────┐
                │  Per-request token streams  │  one async queue per active request
                └────────────┬────────────────┘
                             │ consumed by
                             ▼
                ┌─────────────────────────────┐
                │  HTTP response streamers    │  back to clients
                └─────────────────────────────┘
```

Five concurrent loops that share state. The locking strategy across these layers is what production teams spend serious engineering on.

## Pattern 1 — sharded locks for the KV block manager

The KV block manager (Topic 10's free list + block tables) is a contended resource. Every request that joins or grows touches it.

A single global mutex serializes every allocation. At high concurrency, this becomes the bottleneck.

**vLLM's solution: sharded locks.** Split the block pool into N shards (typically 8 or 16). Each shard has its own lock. Allocations hash to a shard; only that shard's lock is taken.

```python
class ShardedBlockManager:
    def __init__(self, n_blocks: int, n_shards: int = 8):
        self.shards = [
            BlockShard(blocks=range(i, n_blocks, n_shards))
            for i in range(n_shards)
        ]

    def allocate(self, request_id: int) -> int:
        # Hash request_id to a shard (deterministic; a request always hits the same shard)
        shard = self.shards[request_id % len(self.shards)]
        with shard.lock:
            return shard.allocate()
```

Result: 8× the parallelism on allocation contention, modulo hash skew.

Read `vllm/core/block/block_manager_v2.py` source to see this in production code.

## Pattern 2 — single-writer admission queue

The scheduler is the only writer to active batch state. HTTP handlers are *readers* (they observe the batch but don't modify it).

The single-writer / multi-reader pattern is the cleanest concurrency model when it fits:

- No writer-writer races (only one writer)
- Readers can use lock-free reads or RW locks
- No coordination overhead between writes

vLLM's V1 scheduler enforces this — only the scheduler thread writes to the persistent batch; everyone else reads.

## Pattern 3 — cancellation propagation

Client disconnects mid-stream. The decode slot must be freed *promptly*, not when the request hits its `max_tokens` limit.

Naive: poll on every step ("did the client disconnect?"). Wastes CPU and adds latency.

Better: an `asyncio.Event` or `tokio::sync::watch` channel watched by both the HTTP handler and the scheduler:

```python
class RequestState:
    def __init__(self):
        self.cancel_event = asyncio.Event()  # set by HTTP handler on disconnect
        self.token_queue = asyncio.Queue()   # filled by scheduler
        # ...

# In HTTP handler:
async def stream_response(request_state):
    try:
        async for token in request_state.token_stream():
            yield token
    except asyncio.CancelledError:
        request_state.cancel_event.set()  # tell the scheduler
        raise

# In scheduler:
def check_for_cancellations(self):
    for req in self.active_requests:
        if req.cancel_event.is_set():
            self.kv_cache.free_request(req.id)
            self.active_requests.remove(req)
```

Worst case without proper cancellation: zombie decode slot for the full max_tokens (often 256-2048 tokens of wasted compute).

## Pattern 4 — stream multiplexing (per-request token queues)

Each active request has its own bounded async queue of decoded tokens. The scheduler pushes; the HTTP handler pops.

```python
class RequestState:
    def __init__(self):
        self.token_queue = asyncio.Queue(maxsize=64)  # bounded → backpressure
        self.cancel_event = asyncio.Event()
```

When the client is slow (network buffering, etc.), the queue fills, and the scheduler stalls *just that request* — not the whole batch.

This is the right place to apply backpressure: per-stream, not batch-wide.

## Pattern 5 — async vs threaded for tokenization/detokenization

Tokenization is CPU-bound. Running it inline in an async event loop blocks all other coroutines for the duration.

Right pattern: push tokenization to a thread pool:

```python
async def handle_request(prompt: str):
    # Don't block the event loop on tokenize
    token_ids = await asyncio.to_thread(tokenizer.encode, prompt)
    # ...
```

Or, even better, a *separate process* for tokenization (avoids GIL contention with the scheduler thread).

## What to read in vLLM source

vLLM v1's source codifies all these patterns. Worth reading once for the structure:

- `vllm/core/scheduler.py` — the scheduling loop
- `vllm/core/block/block_manager_v2.py` — sharded locks
- `vllm/engine/async_llm_engine.py` — the asyncio bridge to the scheduler
- `vllm/engine/output_processor/` — per-request output stream demux

## Pitfalls

1. **Single global lock around everything.** Naive but common. Replace with sharded or per-resource locks.
2. **Forgetting cancellation.** Zombie decode slots accumulate in production. Test with deliberate disconnects.
3. **Unbounded queues.** A slow client fills the queue, OOM eventually. Always bound queues; let them apply backpressure.
4. **Mixing CPU-bound work into the event loop.** Tokenization, hashing for prefix cache, JSON serialization — all need thread offload.
5. **Race conditions between block allocation and attention compute.** The scheduler must not free a block while a kernel is reading from it. Coordinate via reference counts and step-boundary cleanup.

## What you'll do

Extend your `mini-vllm`:

1. **Sharded block manager.** Replace your single-mutex block manager from Topic 10 with 8 shards. Run a multi-threaded benchmark — confirm tail latency improves.
2. **Cancellation propagation.** Add `asyncio.Event` per request. HTTP handler sets it on disconnect; scheduler checks at step boundary.
3. **Per-stream queues.** Instead of returning all tokens via the future, push tokens to a bounded queue. HTTP handler pops and yields via SSE.
4. **Move tokenization off the event loop.** Use `asyncio.to_thread` or a worker process.

Test:

- Burst 100 requests, kill 30% of clients halfway through. Verify zombie slots are freed within 1 step.
- Slow client (read 1 token/sec) — verify only that one request stalls, not the whole batch.
- Profile lock contention with sharded vs single-mutex block manager.

## References

- vLLM source: `vllm/core/scheduler.py`, `vllm/core/block/block_manager_v2.py`
- vLLM V1 design — https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1
- vLLM cache salting RFC (multi-tenant isolation) — https://github.com/vllm-project/vllm/issues/16016
- SGLang Q2 2026 roadmap (stateless scheduler refactor) — https://github.com/sgl-project/sglang/issues/22949
- Tokio async tutorial — https://tokio.rs/tokio/tutorial (for the Rust analog)
