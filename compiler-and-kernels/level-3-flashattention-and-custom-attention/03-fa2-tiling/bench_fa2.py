"""Benchmark fa2_triton vs F.scaled_dot_product_attention.

Run on a GPU:
    python bench_fa2.py
"""
from __future__ import annotations

import torch
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from fa2_triton import fa2_forward_triton


def bench(shape, dtype=torch.bfloat16, causal=False):
    B, H, N, D = shape
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    def run_sdpa_math():
        with sdpa_kernel(SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    def run_sdpa_flash():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    def run_ours():
        o, _ = fa2_forward_triton(q, k, v, causal=causal)
        return o

    ms_math = triton.testing.do_bench(run_sdpa_math, warmup=25, rep=100)
    try:
        ms_flash = triton.testing.do_bench(run_sdpa_flash, warmup=25, rep=100)
    except Exception:
        ms_flash = float("nan")
    ms_ours = triton.testing.do_bench(run_ours, warmup=25, rep=100)

    # Approx 4 * B * H * N * N * D flops (two matmuls).
    flops = 4 * B * H * N * N * D
    tflops_math = flops / ms_math / 1e9
    tflops_flash = flops / ms_flash / 1e9 if ms_flash == ms_flash else float("nan")
    tflops_ours = flops / ms_ours / 1e9

    print(f"\nshape (B,H,N,D)={shape} dtype={dtype} causal={causal}")
    print(f"  SDPA(MATH):  {ms_math:7.3f} ms   {tflops_math:7.1f} TFLOPs/s")
    print(f"  SDPA(FLASH): {ms_flash:7.3f} ms   {tflops_flash:7.1f} TFLOPs/s")
    print(f"  ours:        {ms_ours:7.3f} ms   {tflops_ours:7.1f} TFLOPs/s   "
          f"({ms_flash/ms_ours:.2f}x vs FLASH)")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    for shape in [(4, 8, 1024, 64), (4, 8, 2048, 64), (2, 8, 4096, 64), (2, 8, 4096, 128)]:
        bench(shape, dtype=torch.bfloat16, causal=False)
        bench(shape, dtype=torch.bfloat16, causal=True)


if __name__ == "__main__":
    main()
