# notes — flashinfer

## Why paged KV is the right layout for a server

- **No fragmentation.** A request that ends frees its pages back to a global pool. The next request claims them. Memory utilization stays near 100%.
- **No memcpy on append.** Token generation grows the last page in place. When the last page fills, allocate a new one — O(1), no copy.
- **Independent of request schedule.** Pages don't have to be contiguous. Requests interleave freely; the block table threads them.

The cost: every K/V load goes through a page-table indirection. ~5–10% slower than contiguous KV per-tile. The win on memory utilization (vs reserving max-context-length contiguous per request) is enormous — vLLM measured 60–80% memory waste in pre-paged engines.

## The dispatch chain I observed

(fill after running vllm_dispatch_trace.py)

- Prefill backend: ___
- Decode backend: ___
- Hardware: ___

## What confused me

- `plan()` vs `run()` separation. The plan computes the tile schedule and stores it in the workspace buffer. `run()` reads the schedule and dispatches. If you change shapes without replanning, the kernel happily uses the stale schedule and you get wrong outputs with no warning.
- `int32` everywhere. CUDA kernels do not like `int64` indices in tight loops. FlashInfer enforces this explicitly.
- The JIT compile is *per attention variant*. If your model has both prefill and decode and uses sliding window, that's 2 variants × 2 backends = 4 first-call JIT compiles. After warm-up they're all cached.

## After this

I can read vLLM's `vllm/v1/attention/backends/flashinfer.py` and follow what each call is doing. I know what page size means in production. I can decide between FlashInfer and pure FlexAttention for a custom variant based on whether I need ragged batching (FlashInfer) or training (FlexAttention).
