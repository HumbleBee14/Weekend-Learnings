"""Benchmark Pattern A vs Pattern B inside a compiled block.

Pattern A should win by 5-15% on bandwidth-bound shapes because Inductor
fuses the residual add into the rmsnorm kernel. Pattern B keeps them as
separate kernels (one HBM round-trip extra).
"""

from __future__ import annotations

import time

import torch

# Import both patterns (registers the ops)
import triton_op_pattern  # noqa: F401
import custom_op_pattern  # noqa: F401

from triton_op_pattern import rmsnorm as rmsnorm_op_a


def bench(fn, *args, n_warm: int = 10, n: int = 100) -> float:
    for _ in range(n_warm):
        y = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def main() -> None:
    assert torch.cuda.is_available()
    dtype = torch.bfloat16
    M, N = 2048, 4096
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.ones(N, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)

    def block_eager(x, w, residual):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return (x.float() * rms * w.float()).to(x.dtype) + residual

    def block_a(x, w, residual):
        return rmsnorm_op_a(x, w, 1e-6) + residual

    def block_b(x, w, residual):
        return torch.ops.level2.rmsnorm_opaque(x, w, 1e-6) + residual

    eager_compiled = torch.compile(block_eager, fullgraph=True)
    a_compiled = torch.compile(block_a, fullgraph=True)
    b_compiled = torch.compile(block_b, fullgraph=True)

    rows = [
        ("eager (uncompiled)", block_eager),
        ("eager compiled", eager_compiled),
        ("Pattern A (triton_op)", a_compiled),
        ("Pattern B (custom_op)", b_compiled),
    ]

    print(f"{'variant':30s}  {'ms/iter':>10s}")
    print("-" * 44)
    for name, fn in rows:
        ms = bench(fn, x, w, residual)
        print(f"{name:30s}  {ms:10.4f}")


if __name__ == "__main__":
    main()
