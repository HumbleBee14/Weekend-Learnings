"""
A standalone naive KV cache implementation. Pre-allocates a max-sized contiguous tensor
per layer per request. Demonstrates the four problems from CONCEPTS.md:

  1. Memory waste from over-allocation
  2. Internal fragmentation
  3. No prefix sharing
  4. Hard upper bound on sequence length

Run:
    pip install torch
    python naive_kv_cache.py
"""

import torch


class NaiveKVCache:
    """
    Pre-allocates one (max_batch, max_seq_len, n_heads, head_dim) tensor per layer
    for K and V each. Tracks the current "fill position" per request.

    Memory: O(max_batch × max_seq_len × n_layers × n_heads × head_dim × 2 (K,V) × dtype_size)

    For Qwen2.5-0.5B (24 layers, 14 heads, 64 head_dim) and max_batch=8, max_seq_len=8192:
       8 × 8192 × 24 × 14 × 64 × 2 × 2 (BF16) = 705 MB
    Whether the requests use that much or not.
    """

    def __init__(
        self,
        max_batch: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        # k_cache and v_cache are lists indexed by layer
        self.k_cache = [
            torch.zeros((max_batch, max_seq_len, n_heads, head_dim), dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v_cache = [
            torch.zeros((max_batch, max_seq_len, n_heads, head_dim), dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        # Per-request fill position
        self.positions = torch.zeros(max_batch, dtype=torch.long, device=device)
        # Which slots are occupied
        self.occupied = torch.zeros(max_batch, dtype=torch.bool, device=device)

    def allocate_slot(self) -> int:
        """Find a free slot. Raises if none."""
        for i in range(self.max_batch):
            if not self.occupied[i]:
                self.occupied[i] = True
                self.positions[i] = 0
                # Zero out (could also leave stale; attention masks should hide it)
                for L in range(self.n_layers):
                    self.k_cache[L][i].zero_()
                    self.v_cache[L][i].zero_()
                return i
        raise RuntimeError("No free slots — bump max_batch or wait for completion")

    def free_slot(self, slot_idx: int) -> None:
        self.occupied[slot_idx] = False
        self.positions[slot_idx] = 0

    def append(self, layer: int, slot_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append a single token's K, V to the cache at the next position."""
        pos = self.positions[slot_idx].item()
        if pos >= self.max_seq_len:
            raise RuntimeError(f"Sequence exceeds max_seq_len={self.max_seq_len}")
        # k, v shapes: (n_heads, head_dim) for one token
        self.k_cache[layer][slot_idx, pos] = k
        self.v_cache[layer][slot_idx, pos] = v
        # Only the LAST layer's append should bump position (each layer is one append per token)
        if layer == self.n_layers - 1:
            self.positions[slot_idx] += 1

    def read(self, layer: int, slot_idx: int):
        """Return the populated K, V slices for this slot at this layer."""
        pos = self.positions[slot_idx].item()
        return self.k_cache[layer][slot_idx, :pos], self.v_cache[layer][slot_idx, :pos]

    def memory_used_mb(self) -> float:
        """Total memory reserved by this cache, regardless of utilization."""
        total = 0
        for c in self.k_cache + self.v_cache:
            total += c.element_size() * c.numel()
        return total / 1e6

    def memory_actually_used_mb(self) -> float:
        """Memory for the slots actually being used."""
        used = 0
        for slot in range(self.max_batch):
            if self.occupied[slot]:
                pos = self.positions[slot].item()
                # Per-token size = 2 (K,V) × n_layers × n_heads × head_dim × dtype_size
                per_token = self.k_cache[0].element_size() * self.k_cache[0].shape[2] * self.k_cache[0].shape[3]
                used += pos * 2 * self.n_layers * per_token
        return used / 1e6


def main():
    if not torch.cuda.is_available():
        print("CUDA recommended; running on CPU for demo.")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.bfloat16

    # Qwen2.5-0.5B-ish dimensions
    cache = NaiveKVCache(
        max_batch=8,
        max_seq_len=8192,
        n_layers=24,
        n_heads=14,
        head_dim=64,
        dtype=dtype,
        device=device,
    )

    print(f"Created cache. Reserved memory: {cache.memory_used_mb():.0f} MB")
    print(f"This is reserved REGARDLESS of how many tokens we actually use.\n")

    # Allocate 3 slots, fill with mixed lengths
    slots_and_tokens = [(0, 100), (1, 200), (2, 5000)]
    for slot_idx, n_tokens in slots_and_tokens:
        cache.allocate_slot()
        # Simulate filling
        for _ in range(n_tokens):
            for layer in range(cache.n_layers):
                k = torch.randn(14, 64, dtype=dtype, device=device)
                v = torch.randn(14, 64, dtype=dtype, device=device)
                cache.append(layer=layer, slot_idx=slot_idx, k=k, v=v)

    print(f"Allocated 3 slots with [100, 200, 5000] tokens.")
    print(f"Memory actually used:  {cache.memory_actually_used_mb():.0f} MB")
    print(f"Memory reserved total: {cache.memory_used_mb():.0f} MB")
    print(f"Wasted:                {cache.memory_used_mb() - cache.memory_actually_used_mb():.0f} MB")
    util = cache.memory_actually_used_mb() / cache.memory_used_mb() * 100
    print(f"Utilization:           {util:.1f}%")
    print()
    print("Pain points to feel here:")
    print("  1. Memory waste — 5 unused slots, plus padding within used slots")
    print("  2. Hard cap — request #4 with 9000 tokens would crash (max_seq_len=8192)")
    print("  3. No prefix sharing — slot 0 and slot 1's first 100 tokens are recomputed")
    print("  4. Once a slot is freed, the gap is not reusable until the surrounding traffic clears")
    print()
    print("These pain points motivate paged KV cache (Topic 10). vLLM's PagedAttention solves all four.")


if __name__ == "__main__":
    main()
