# 10 — Paged KV Cache

## Files

- `CONCEPTS.md` — the OS virtual memory analogy, what paging fixes from Topic 09's naive cache, the components (block pool, free list, block tables, attention kernel)
- `paged_kv_cache.py` — a working paged KV cache with allocate/append/free/gather. Run the same workload from Topic 09 and compare.

## Quickstart

```bash
pip install torch
python paged_kv_cache.py
```

## Expected output

```
Created paged KV cache: 512 blocks × 16 tokens
Total token capacity: 8192

After 3 requests with [100, 200, 5000] tokens:
  Blocks in use:        332 / 512
  Block utilization:    64.8%
  Actual tokens:        5300
  Reserved token slots: 5312
  Slot utilization:     99.8%

Adding a 4th request with 9000 tokens (would crash naive cache)...
  Pool exhausted: Out of blocks
  This is the only failure mode of paged KV — the pool itself.

Freeing request 0...
  Free blocks now: 187
  These can be claimed by ANY new or growing request immediately.
```

The 99.8% slot utilization vs 65% in Topic 09 is the headline. Same workload, much higher density.

## Try

- **Increase pool size to 1024 blocks**. Now 9000-token request fits.
- **Free a request mid-stream**. Confirm freed blocks return to the pool.
- **Run two requests with the same prompt**. Both will compute and store the prefix's KV — *no prefix sharing yet*. Topic 11 covers prefix caching, the first big optimization on top of paging.
- **Stress test**: 100 small requests followed by one big one. Watch utilization climb.

## Connection to FlashInfer

The KV cache *storage* you built here is the right structure. The *attention kernel* that reads from it efficiently is non-trivial — that's where FlashInfer's page-table attention comes in.

In production, you don't write a custom attention kernel for paged KV. You use FlashInfer:

```python
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

wrapper = BatchPrefillWithPagedKVCacheWrapper(...)
wrapper.begin_forward(qo_indptr, paged_kv_indptr, paged_kv_indices, ...)
output = wrapper.forward(q, paged_kv_cache)
```

The block table goes in; an efficient attention computation comes out. vLLM, SGLang, TRT-LLM all use this.

## What you should walk away with

- Working paged KV cache that handles mixed-length workloads cleanly
- Concrete numbers showing 1.5× memory utilization gain over naive
- Understanding of the components (pool, free list, block tables, page-table attention)
- The setup for Topic 11 (eviction policies and prefix caching) and Topic 12 (long-context stress)

## Where this goes

- Topic 11 — eviction policies (LRU, sliding window, prefix sharing)
- Topic 12 — stress test with 100K+ token sequences
- Together they finish the KV cache sub-arc and `mini-vllm` becomes recognizable as a real serving engine
