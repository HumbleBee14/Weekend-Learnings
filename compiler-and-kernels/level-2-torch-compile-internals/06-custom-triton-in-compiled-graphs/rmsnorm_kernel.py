"""Shared Triton RMSNorm kernel. One-pass online stats; row-per-program.

This is the same shape as the Level 1 sub-module 03 'single-pass with online
stats' version, kept simple for readability. Production code would autotune
BLOCK_SIZE.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_x_row,
    stride_y_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = X_ptr + row * stride_x_row
    y_row_ptr = Y_ptr + row * stride_y_row

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    rrms = tl.rsqrt(sumsq / N + eps)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * rrms) * w
    tl.store(y_row_ptr + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def launch_rmsnorm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Drive the kernel. Assumes x is (M, N) row-major."""
    assert x.is_cuda and w.is_cuda and out.is_cuda
    M, N = x.shape
    BLOCK = next_pow2(N)
    rmsnorm_kernel[(M,)](
        x, w, out,
        x.stride(0), out.stride(0),
        N=N, eps=eps, BLOCK_SIZE=BLOCK,
    )
