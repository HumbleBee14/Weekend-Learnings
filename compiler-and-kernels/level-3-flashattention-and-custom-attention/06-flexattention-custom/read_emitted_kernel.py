"""Dump the Inductor-emitted Triton for a FlexAttention call. Inspect what was inlined.

Run:
    TORCH_LOGS="output_code" python read_emitted_kernel.py 2> kernel_dump.py
    # then open kernel_dump.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_LOGS", "output_code")

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)
    B, H, N, D = 1, 4, 1024, 64

    def sliding_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= 256)

    block_mask = create_block_mask(sliding_causal, B=None, H=None, Q_LEN=N, KV_LEN=N, device="cuda")

    flex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

    # First call compiles and (with TORCH_LOGS=output_code) prints the kernel to stderr.
    out = flex(q, k, v, block_mask=block_mask)
    print("Compiled and ran. Search the emitted code for:")
    print("  - the inner KV loop")
    print("  - mask_mod logic (q_idx - kv_idx <= 256)")
    print("  - block-skip / partial-block dispatch")
    print(f"out.shape = {out.shape}")


if __name__ == "__main__":
    main()
