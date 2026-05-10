# 11 — KV Cache Eviction & Prefix Sharing

## Files

- `CONCEPTS.md` — when paging runs out, prefix sharing as the first optimization, eviction policies (LRU / sliding-window / priority / prefix-aware), the block-hash kv-connector standard, cache salting

## What you do this topic

Extend your paged KV cache from Topic 10:

1. Add a prefix-hash table that maps `prefix_hash → list[block_ids]`
2. Add reference counting on blocks (so shared prefixes aren't double-freed)
3. Add an LRU eviction policy when the pool fills up
4. Test with workloads that have shared system prompts

This is the second-to-last KV-cache step before `mini-vllm` is real.

## Reading-driven topic

Most of the value is in CONCEPTS.md. The implementation pattern is straightforward (hash → check → share or allocate; ref-count up/down; LRU evict when full). The interesting part is the *design choices*:

- Hash chain (block_n's hash includes block_n-1's hash)
- SHA-256 specifically (overkill but standard; safe against collisions across millions of requests)
- Per-tenant salting for multi-tenant safety
- Prefix-aware eviction (don't evict shared blocks even if "old")

## Quick proof of concept

Modify your Topic 10 cache to add a prefix table:

```python
class PagedKVCacheWithPrefixSharing(PagedKVCache):
    def __init__(self, ...):
        super().__init__(...)
        # Hash → list of block IDs, plus refcount
        self.prefix_blocks: dict[bytes, list[int]] = {}
        self.refcount: dict[int, int] = {b: 0 for b in range(self.n_blocks)}

    def allocate_request_with_prefix_check(self, request_id: int, prompt_token_ids: list[int]) -> int:
        """Return number of tokens that hit the prefix cache (already computed)."""
        # Hash blocks of 16 tokens at a time, with chain
        block_hashes = self._compute_block_hash_chain(prompt_token_ids)
        
        block_ids = []
        prefix_hits = 0
        for h in block_hashes:
            if h in self.prefix_blocks:
                # Hit — reuse
                shared = self.prefix_blocks[h][0]  # one block per hash
                block_ids.append(shared)
                self.refcount[shared] += 1
                prefix_hits += 16
            else:
                # Miss — allocate, fill later, register
                new_block = self.free_blocks.popleft()
                block_ids.append(new_block)
                self.refcount[new_block] = 1
                self.prefix_blocks[h] = [new_block]
        
        self.block_tables[request_id] = block_ids
        self.lengths[request_id] = len(prompt_token_ids)
        return prefix_hits
```

Now run two requests with the same 100-token system prompt. The second's prefix_hits should be ~96 (six blocks shared, the seventh is partial and request-specific).

## Try

- **Run 100 requests sharing a system prompt.** Memory should stay roughly constant (only one copy of the prompt's blocks live in the pool).
- **Run 100 requests with unique prefixes.** Memory grows linearly. Eventually the pool fills and you need eviction.
- **Implement LRU.** When `popleft()` fails, evict the oldest unreferenced block.
- **Compare to vLLM**. Run the same workload against vLLM with prefix caching on. Confirm your numbers match in shape (you'll be slower; production code is well-tuned).

## Where this goes

Topic 12 is the final KV-cache step: stress test with 100K+ token sequences. With paging + prefix sharing + LRU eviction, your `mini-vllm` should handle long-context workloads that would crash a naive implementation.

After Topic 12, the KV-cache sub-arc is done. The remaining topics in Level 4 are speculative decoding (13, 17), continuous batching (14), structured output (15), and serving concurrency (16).
