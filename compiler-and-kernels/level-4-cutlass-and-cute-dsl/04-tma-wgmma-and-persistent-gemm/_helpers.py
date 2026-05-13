"""
_helpers.py — shared benchmarking and correctness helpers for stage1..stage5.

Each stage imports `bench` and `verify` from here. The cuBLAS reference is
just `torch.matmul`, which routes to cuBLAS for BF16 on Hopper.
"""

from __future__ import annotations

import torch


def cublas_tflops(M: int, N: int, K: int, dtype: torch.dtype = torch.bfloat16,
                  n_iter: int = 100) -> tuple[float, float]:
    """Measure cuBLAS reference TFLOPS for the same shape."""
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    for _ in range(25):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / n_iter
    flops = 2.0 * M * N * K
    tflops = flops / (ms * 1e9)
    return ms, tflops


def bench(fn, n_iter: int = 100, n_warmup: int = 25) -> float:
    """Return mean ms per call."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def verify(actual: torch.Tensor, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-2):
    expected = torch.matmul(a.float(), b.float()).to(actual.dtype)
    max_diff = (actual.float() - expected.float()).abs().max().item()
    assert max_diff < atol, f"max diff {max_diff} exceeds {atol}"


def report(name: str, M: int, N: int, K: int, ms: float, dtype: str = "bf16"):
    flops = 2.0 * M * N * K
    tflops = flops / (ms * 1e9)
    _, cublas = cublas_tflops(M, N, K)
    pct = 100.0 * tflops / cublas
    print(f"{name:30s}  {M}x{N}x{K} {dtype}  {ms:7.3f} ms  "
          f"{tflops:7.1f} TFLOPS  ({pct:5.1f}% cuBLAS)")
