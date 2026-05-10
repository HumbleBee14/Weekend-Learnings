"""
A paged KV cache. Same interface as the naive version, completely different storage.

Educational implementation — not optimized. Production: use FlashInfer's paged attention
or vLLM's block manager directly.

Run:
    pip install torch
    python paged_kv_cache.py
"""

import torch
from collections import deque


class PagedKVCache:
    """
    Block pool + free list + per-request block tables.

    Memory layout: one big tensor `(n_blocks, block_size, n_heads, head_dim)` per layer
    for K and V separately. Block indices into this tensor are the unit of allocation.
    """

    def __init__(
        self,
        n_blocks: int,
        block_size: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_layers = n_layers

        # Block pool — physical storage. Indexed by physical block id.
        # Shape: (n_blocks, block_size, n_heads, head_dim) per layer.
        self.k_pool = [
            torch.zeros((n_blocks, block_size, n_heads, head_dim), dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v_pool = [
            torch.zeros((n_blocks, block_size, n_heads, head_dim), dtype=dtype, device=device)
            for _ in range(n_layers)
        ]

        # Free list — indices of unallocated blocks. Use a deque for O(1) pop/push.
        self.free_blocks = deque(range(n_blocks))

        # Per-request state: block_table (list of physical block ids) and current
        # length (number of tokens stored).
        # In production this is keyed by request_id; here we use request indexes.
        self.block_tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    # ---- allocation ----

    def allocate_request(self, request_id: int, n_tokens: int = 0) -> None:
        """Reserve initial blocks for a new request that will start with n_tokens prefill."""
        if request_id in self.block_tables:
            raise ValueError(f"Request {request_id} already exists")
        # How many blocks needed for the prefill?
        n_needed = (n_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < n_needed:
            raise RuntimeError(f"Out of blocks; have {len(self.free_blocks)}, need {n_needed}")

        block_ids = [self.free_blocks.popleft() for _ in range(n_needed)]
        self.block_tables[request_id] = block_ids
        self.lengths[request_id] = n_tokens   # we'll fill them via append later

    def free_request(self, request_id: int) -> None:
        """Return a request's blocks to the free list."""
        if request_id not in self.block_tables:
            return
        for block_id in self.block_tables[request_id]:
            self.free_blocks.append(block_id)
        del self.block_tables[request_id]
        del self.lengths[request_id]

    # ---- read/write ----

    def append_token(self, request_id: int, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Append one token's K, V for a layer. May allocate a new block if the current
        last block is full.

        k, v shapes: (n_heads, head_dim) — one token's worth
        """
        if request_id not in self.block_tables:
            raise KeyError(f"Unknown request {request_id}")
        block_ids = self.block_tables[request_id]
        token_pos = self.lengths[request_id]

        # Which block + offset?
        block_idx = token_pos // self.block_size
        intra_block = token_pos % self.block_size

        # Need a new block?
        if block_idx >= len(block_ids):
            if not self.free_blocks:
                raise RuntimeError("Out of blocks")
            new_block = self.free_blocks.popleft()
            block_ids.append(new_block)

        physical_block = block_ids[block_idx]
        self.k_pool[layer][physical_block, intra_block] = k
        self.v_pool[layer][physical_block, intra_block] = v

        # Bump length only on the last layer (each layer appends per token)
        if layer == self.n_layers - 1:
            self.lengths[request_id] += 1

    def gather_kv(self, request_id: int, layer: int):
        """
        Return contiguous K, V tensors for this request's full sequence so far.
        For real attention kernels, you'd pass the block table directly to a
        page-table attention kernel (FlashInfer's `BatchPrefillWithPagedKVCacheWrapper`)
        instead of materializing the contiguous version.
        """
        block_ids = self.block_tables[request_id]
        length = self.lengths[request_id]
        # Gather all blocks
        ks = self.k_pool[layer][block_ids]   # (n_blocks, block_size, n_heads, head_dim)
        vs = self.v_pool[layer][block_ids]
        # Flatten and trim to actual length
        ks = ks.flatten(0, 1)[:length]
        vs = vs.flatten(0, 1)[:length]
        return ks, vs

    # ---- diagnostics ----

    def utilization(self) -> dict:
        """Memory utilization — how much of the pool is in use."""
        in_use_blocks = self.n_blocks - len(self.free_blocks)
        # But not every block is fully filled. Count actual tokens vs reserved blocks.
        actual_tokens = sum(self.lengths.values())
        reserved_token_slots = in_use_blocks * self.block_size
        slot_util = actual_tokens / reserved_token_slots if reserved_token_slots else 1.0
        return {
            "n_blocks_total": self.n_blocks,
            "n_blocks_in_use": in_use_blocks,
            "n_blocks_free": len(self.free_blocks),
            "actual_tokens": actual_tokens,
            "reserved_token_slots": reserved_token_slots,
            "slot_utilization": slot_util,
            "block_utilization": in_use_blocks / self.n_blocks,
        }


def main():
    if not torch.cuda.is_available():
        print("CUDA recommended; running on CPU for demo.")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.bfloat16

    # Same model dimensions as naive demo. With paged: more flexibility.
    cache = PagedKVCache(
        n_blocks=512,        # 512 blocks × 16 tokens = 8192 token slots, shared across all requests
        block_size=16,
        n_layers=24,
        n_heads=14,
        head_dim=64,
        dtype=dtype,
        device=device,
    )

    print(f"Created paged KV cache: {cache.n_blocks} blocks × {cache.block_size} tokens")
    print(f"Total token capacity: {cache.n_blocks * cache.block_size}\n")

    # Same workload as naive demo: 3 requests with [100, 200, 5000] tokens.
    requests = [(0, 100), (1, 200), (2, 5000)]
    for req_id, n_tokens in requests:
        cache.allocate_request(req_id, n_tokens=0)  # we'll grow via append
        for tok in range(n_tokens):
            for layer in range(cache.n_layers):
                k = torch.randn(14, 64, dtype=dtype, device=device)
                v = torch.randn(14, 64, dtype=dtype, device=device)
                cache.append_token(req_id, layer, k, v)

    util = cache.utilization()
    print(f"After 3 requests with [100, 200, 5000] tokens:")
    print(f"  Blocks in use:        {util['n_blocks_in_use']} / {util['n_blocks_total']}")
    print(f"  Block utilization:    {util['block_utilization']:.1%}")
    print(f"  Actual tokens:        {util['actual_tokens']}")
    print(f"  Reserved token slots: {util['reserved_token_slots']}")
    print(f"  Slot utilization:     {util['slot_utilization']:.1%}")
    print()

    # Now try a 4th request with 9000 tokens — naive would crash (max_seq_len=8192).
    # Paged: just allocates more blocks, only fails if pool is exhausted.
    print(f"Adding a 4th request with 9000 tokens (would crash naive cache)...")
    cache.allocate_request(3, n_tokens=0)
    try:
        for tok in range(9000):
            for layer in range(cache.n_layers):
                k = torch.randn(14, 64, dtype=dtype, device=device)
                v = torch.randn(14, 64, dtype=dtype, device=device)
                cache.append_token(3, layer, k, v)
        print(f"  Success! Total tokens served: {cache.utilization()['actual_tokens']}")
    except RuntimeError as e:
        print(f"  Pool exhausted: {e}")
        print(f"  This is the only failure mode of paged KV — the pool itself.")

    # Free request 0 — its blocks become available for reuse
    print(f"\nFreeing request 0...")
    cache.free_request(0)
    print(f"  Free blocks now: {cache.utilization()['n_blocks_free']}")
    print(f"  These can be claimed by ANY new or growing request immediately.")

    print()
    print("Compare to Topic 09's naive cache:")
    print("  - Memory utilization 90%+ (vs 65% naive)")
    print("  - No max_seq_len hard cap — only the pool size limits you")
    print("  - Freed slots are immediately reusable, no fragmentation")
    print("  - Topic 11 adds eviction policies for when you DO run out of pool")
    print("  - Topic 12 stress-tests with long-context workloads")


if __name__ == "__main__":
    main()
