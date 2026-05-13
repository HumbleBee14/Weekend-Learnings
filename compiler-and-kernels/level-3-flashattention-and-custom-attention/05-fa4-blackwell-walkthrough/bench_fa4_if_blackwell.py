"""Run FA4 on B200 and compare. Optional; requires SM100 hardware and the
flash-attn CuTeDSL build.

Run:
    pip install -U flash-attn  # or build from source per the FA4 README
    python bench_fa4_if_blackwell.py
"""
from __future__ import annotations

import math

import torch
import triton


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        raise SystemExit(f"Need SM100+ (Blackwell) for FA4. Got cc={cap}.")

    try:
        from flash_attn.cute import flash_attn_func as fa4  # FA4 CuTeDSL entry point
    except ImportError:
        raise SystemExit("FA4 CuTeDSL not importable. Install flash-attn from source.")

    torch.manual_seed(0)
    B, H, N, D = 2, 16, 8192, 128
    q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

    # Approx 4 * B * H * N * N * D flops.
    flops = 4 * B * H * N * N * D

    ms_fa4 = triton.testing.do_bench(lambda: fa4(q, k, v, causal=True), warmup=25, rep=100)

    # cuDNN attention for reference.
    q_bhnd = q.permute(0, 2, 1, 3).contiguous()
    k_bhnd = k.permute(0, 2, 1, 3).contiguous()
    v_bhnd = v.permute(0, 2, 1, 3).contiguous()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        ms_cudnn = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q_bhnd, k_bhnd, v_bhnd, is_causal=True
            ),
            warmup=25, rep=100,
        )

    # Causal halves the work approximately.
    eff_flops = flops * 0.5
    print(f"shape (B,H,N,D)=({B},{H},{N},{D}) bf16 causal=True")
    print(f"  FA4 (CuTeDSL):  {ms_fa4:7.3f} ms   {eff_flops/ms_fa4/1e9:7.1f} TFLOPs/s")
    print(f"  cuDNN attn:     {ms_cudnn:7.3f} ms   {eff_flops/ms_cudnn/1e9:7.1f} TFLOPs/s")
    print(f"  ratio FA4/cuDNN: {ms_cudnn/ms_fa4:.2f}x")


if __name__ == "__main__":
    main()
