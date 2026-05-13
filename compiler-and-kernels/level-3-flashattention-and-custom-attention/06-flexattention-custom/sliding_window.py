"""Sliding-window causal attention via mask_mod + BlockMask. Compare speedup to sparsity.

Run on GPU:
    python sliding_window.py
"""
from __future__ import annotations

import torch
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)
    B, H, D = 2, 8, 64
    WINDOW = 1024

    flex = torch.compile(flex_attention, dynamic=False)

    print(f"{'N':>6} {'sparsity':>10} {'sdpa ms':>10} {'flex ms':>10} {'speedup':>10}")
    for N in [2048, 4096, 8192, 16384]:
        def sliding_causal(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx <= WINDOW)

        block_mask = create_block_mask(sliding_causal, B=None, H=None, Q_LEN=N, KV_LEN=N, device="cuda")
        # block_mask.sparsity() returns fraction *kept*. Convert.
        kept = block_mask.sparsity()  # PyTorch 2.5+: this method name; if not, compute from num_blocks/total

        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            try:
                ms_sdpa = triton.testing.do_bench(
                    lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
                    warmup=25, rep=100,
                )
            except Exception:
                ms_sdpa = float("nan")

        ms_flex = triton.testing.do_bench(
            lambda: flex(q, k, v, block_mask=block_mask), warmup=25, rep=100
        )

        speedup = ms_sdpa / ms_flex if ms_sdpa == ms_sdpa else float("nan")
        print(f"{N:>6} {kept:>10.4f} {ms_sdpa:>10.3f} {ms_flex:>10.3f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
