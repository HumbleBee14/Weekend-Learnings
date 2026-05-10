# 11 — KV Cache Eviction & Prefix Sharing

## When does a paged KV cache run out?

The pool is finite. With long contexts and many concurrent users, you eventually fill it. Three options:

1. **Reject new requests** — backpressure (Level 7's domain)
2. **Evict** — drop existing requests' KV to make room
3. **Offload** — move cold KV to CPU/disk and reload when needed (LMCache, Topic 12)

This topic covers eviction (option 2) and the optimization that delays the need for it: **prefix sharing**.

## Prefix sharing — the optimization on top of paging

Two requests with the same prompt prefix:

```
Request A: "You are a helpful assistant. Translate to French: 'Hello'"
Request B: "You are a helpful assistant. Translate to French: 'World'"
```

The first 9 tokens are identical. Their KV vectors at every layer are identical. Computing them twice is waste.

### How paging makes this trivial

In the naive cache (Topic 09), prefixes lived in different per-request memory. No way to share.

In the paged cache:
1. Compute the prefix's KV blocks once (when the first request arrives)
2. Hash the prefix tokens to a key
3. Store the prefix's block IDs keyed by that hash
4. When request B arrives, hash its prefix → find the same block IDs → **set request B's block table to point to those same blocks**

Both requests' attention reads the same physical blocks. Memory cost: O(1 prefix). Compute cost: 1 prefill of the prefix.

For shared system prompts (chat assistants), this is enormous. A 4-KB system prompt at the start of every request → 99% prefix hit rate → near-zero per-request prefill cost on that 4 KB.

## The 2026 standard: block-hash kv-connector

The cross-engine standard between vLLM and LMCache. Both compute block hashes the same way:

- Block size 16 tokens
- Hash = SHA-256 of (parent_block_hash, token_ids)
- The hash chain links blocks: hash(block_n) depends on hash(block_n-1)

Why a chain? Because two requests share a prefix only if their *first N blocks are identical AND in the same order*. The hash chain captures both.

Other engines (SGLang, TRT-LLM) hash differently but the principle is the same: each block has a content-derived ID; identical blocks across requests are deduped.

### Cache salting (RFC #16016)

A 2026 addition for security: per-tenant salt mixed into the hash. Prevents tenant A from seeing tenant B's prefixes through cache hits. Important for multi-tenant deployments.

## Eviction policies — when the pool fills up

Once prefix sharing has minimized waste and the pool is *still* full, you have to make room. Policies:

### LRU (Least Recently Used)

Standard. When you need a block, evict the least-recently-touched. Works well for typical chat workloads where idle conversations should be evicted first.

Failure mode: a long-context request that occasionally generates tokens looks "active" forever, never gets evicted, blocks others.

### Sliding window

Each request keeps only the last N tokens of KV. Older tokens are dropped (assumes they're not relevant for future predictions).

Quality cost: the model can't attend to evicted tokens. Most models tolerate this for context >> 4K but quality degrades. Used in production for streaming dialogs (Mistral 7B's sliding window attention is built-in).

### Priority-based

Tag each request with a priority. Evict low-priority requests' blocks first. Real systems use this for:

- Free vs paid tiers (paid never gets evicted)
- Foreground vs background tasks
- Latency-sensitive vs batch jobs

### Prefix-aware

Don't evict prefix blocks that other requests share. If 100 requests share a system prompt's blocks, evicting them would force 100 re-prefills — terrible.

vLLM V1 uses **reference counting**: each block has a refcount. Evict only blocks with refcount=0 (no active request uses them). Falls back to LRU among eligible blocks.

## Pitfalls

1. **Hash collision risk.** SHA-256 is overkill for collision avoidance but cheap enough. Don't use weaker hashes (xxh64, etc.) without thinking about collisions across millions of requests.
2. **Hashing tokens that haven't been processed yet.** The hash should be over input tokens, not generated tokens (those vary per request even with the same prompt).
3. **Forgetting to invalidate on evict.** When you evict a block, remove its hash from the prefix-cache table. Otherwise the next request hashes to it, follows the pointer, and reads garbage.
4. **Sharing across tenants without salting.** Security risk. Use cache salting (RFC #16016) or separate pools per tenant.
5. **Evicting during attention compute.** Same race condition issue as Topic 10. Coordinate with the scheduler.

## Quality measurement for eviction

Sliding-window eviction *changes the model's behavior* — it can no longer attend to dropped tokens. For correctness:

- Test with `lm-eval-harness` (Topic 06) at the eviction window size you'll deploy with
- Compare with-eviction vs without-eviction on long-context tasks
- The drop should be small if window is large enough; verify, don't assume

LRU eviction *doesn't change behavior* — only affects which requests stay live. No quality test needed; it's a scheduling decision.

## What you'll build

Extend your paged KV from Topic 10:

1. Add a `prefix_cache: dict[hash, list[block_ids]]` mapping
2. On allocate: hash the prompt's blocks, check the cache, share existing blocks where possible
3. Reference-count blocks: increment when shared, decrement when a request frees
4. LRU tracking: timestamp each block on access
5. When pool full: evict the oldest unreferenced block
6. Optional: priority field on requests; evict low-priority first

Test:

- Two requests with same prefix → second should reuse the first's blocks; verify by checking block_table contents
- 100 small requests with the same system prompt → memory should be ~constant (only one copy of system prompt's blocks)
- 100 small requests with unique prefixes → memory should grow linearly

## References

- vLLM prefix caching docs — https://docs.vllm.ai/en/latest/usage/prefix_caching.html
- LMCache architecture — https://docs.lmcache.ai/developer_guide/architecture.html
- llm-d KV cache wins — https://llm-d.ai/blog/kvcache-wins-you-can-see
- SGLang RadixAttention paper — https://arxiv.org/abs/2312.07104 (their prefix sharing)
- vLLM cache salting RFC — https://github.com/vllm-project/vllm/issues/16016
