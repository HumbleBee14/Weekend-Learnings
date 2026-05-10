# 10 — Paged KV Cache

## The big idea

Treat KV cache like virtual memory in an operating system.

OS: process sees a contiguous virtual address space. The OS maps it to non-contiguous physical pages.

Paged KV: each request sees a "logical" sequence of tokens. The KV cache manager maps it to non-contiguous physical *blocks* (typically 16 tokens each).

```
Logical view (one request):
  tokens [0..15] → block 0
  tokens [16..31] → block 1
  tokens [32..47] → block 2
  tokens [48..56] → block 3 (partially filled)

Physical layout (shared across all requests):
  Pool of N blocks of 16 tokens each
  Block 47, 12, 304, 88 happen to be allocated to this request
  (in any order — their position in the pool doesn't matter)

Block table for this request:  [47, 12, 304, 88]
```

The block table tells the attention kernel: "to read this request's tokens, fetch blocks 47, 12, 304, 88, in that order."

## What this fixes

The four problems from Topic 09 — all of them:

### Memory waste from over-allocation: GONE

You don't pre-allocate `max_seq_len`. You allocate blocks as needed. A 100-token request uses ~7 blocks; a 5000-token request uses ~313 blocks. Same pool serves both.

### Internal fragmentation: GONE

Blocks have *fixed size*. When a request finishes, its blocks return to the free pool. They're immediately reusable by any other request. No fragmentation.

### Prefix sharing: ENABLED

Two requests with the same first 100 tokens? Compute those 100 tokens' KV once, store in blocks, and **let both requests' block tables point to the same physical blocks** for the shared prefix.

This is the basis of "prefix caching" — the next-best optimization after paging itself. We cover it more in Topic 11.

### Hard upper bound: GONE

Sequence length is limited only by the total pool size, not per-request reservation. Long requests just allocate more blocks.

## The PagedAttention paper

vLLM (Sept 2023, formalized in PagedAttention paper). The original implementation. Now standard across vLLM, SGLang, TensorRT-LLM, FlashInfer.

What changed in 2026:

- **vLLM V1 scheduler rewrite** with persistent batch and diff-based updates
- **Block hash uses SHA-256** by default (configurable), 16-token blocks default
- **Prefix caching block-hash kv-connector** is the emerging cross-engine standard between vLLM and LMCache
- **Cache salting** (RFC #16016) for per-tenant isolation in multi-tenant deployments
- **Multimodal extension** — `mm_hashes` for image tokens

## The components of a paged KV cache

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Block Pool                                                   │
│    Big tensor of shape (n_blocks, block_size, n_heads, head_dim)│
│    For K and V separately (or interleaved).                     │
│    n_blocks ≈ available_GPU_memory / per_block_bytes            │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Free List                                                    │
│    Set of unused block indices.                                 │
│    Allocate: pop one. Free: push back.                          │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Block Tables (one per request)                               │
│    List of physical block indices for that request.             │
│    Updated as the request grows.                                │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Attention Kernel                                             │
│    Takes Q, the block pool, and the block table.                │
│    Gathers K, V from the (non-contiguous) blocks.               │
│    Computes attention.                                          │
│    This is FlashInfer's `paged_attention` kernel.               │
└─────────────────────────────────────────────────────────────────┘
```

The hard part is the attention kernel — gathering non-contiguous blocks efficiently. In 2026 this is a solved problem: FlashInfer's page-table attention is the standard.

## Block size — the lever

```
block_size=1     too small; per-block overhead dominates
block_size=16    vLLM's default; good balance
block_size=32    slightly less overhead, slightly more internal fragmentation
block_size=128   too large; internal fragmentation hurts
```

The trade-off:
- Smaller blocks → less waste at the *end* of sequences (less padding within last block)
- Smaller blocks → more block-table overhead, more attention-kernel indexing cost
- 16 has been the empirical sweet spot since 2023

## What you'll build

A paged KV cache that integrates with `mini-serve`:

1. Allocate one big block pool tensor
2. Free list (just a Python set, or a more efficient data structure for production)
3. Per-request block table (list of block indices)
4. `allocate_request(prompt_length)` → list of blocks for the prompt
5. `append_token(request_id, k, v)` → grab a new block if last one is full
6. `free_request(request_id)` → return blocks to free list
7. Modify the attention kernel to use the block table

For the attention kernel: easiest path is to use FlashInfer's paged attention rather than write your own. Production-grade. Full integration shown in `paged_kv_cache.py`.

## Pitfalls

1. **Forgetting the partial-block accounting.** When a request has 25 tokens with block_size=16, it uses 2 blocks (one full, one with 9 used / 7 unused). Track this carefully.
2. **Race conditions in the allocator.** Multi-threaded allocation needs locks (Topic 16 covers sharded locks).
3. **Memory exhaustion.** Block pool is finite. When it's full, you must either evict (Topic 11) or reject new requests. Production: usually the latter for fairness.
4. **Block table updates aren't atomic with attention compute.** A request whose block table is being modified while attention is reading it will give wrong answers. Coordinate with the scheduler.
5. **Forgetting to handle FP8/FP4 KV cache.** Block storage type matters. FP8 KV halves memory at small quality cost; FP4 KV quarters at bigger quality cost.

## What you should walk away with

A working paged KV cache. Compare to your naive cache from Topic 09:

- Same hardware budget supports ~3-5× more concurrent requests
- No more hard `max_seq_len` cap
- Mixed-length traffic doesn't waste memory

This is the heart of `mini-vllm`. Once you have paged KV + the eviction policies (Topic 11), you've reimplemented vLLM's core data structure.

## References

- vLLM PagedAttention paper — https://arxiv.org/abs/2309.06180
- vLLM V1 inference request lifecycle — https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1
- FlashInfer paged attention — https://github.com/flashinfer-ai/flashinfer
- vLLM block manager source (csrc) — https://github.com/vllm-project/vllm/tree/main/csrc/cache_kernels
- LMCache architecture — https://docs.lmcache.ai/developer_guide/architecture.html
