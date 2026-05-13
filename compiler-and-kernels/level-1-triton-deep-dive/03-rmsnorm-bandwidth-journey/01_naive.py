"""
Step 01 — Naive RMSNorm.

One program per row. Two passes over the row (one for the squared sum,
one for the normalization). BLOCK_SIZE picked by intuition. No autotune.

Expected: 10–15% of peak HBM bandwidth on whatever GPU you're on.
This is the floor — every step from here adds bandwidth.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_naive_kernel(
    out_ptr, x_ptr, w_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    # PASS 1: load row, compute sum of squares.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    rms = tl.sqrt(sum_sq / n_cols + eps)

    # PASS 2: load row AGAIN, normalize, multiply by weight, store.
    # This second load is wasteful. The next step fixes it.
    x_again = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x_again / rms) * w
    tl.store(out_row + cols, out.to(x.dtype), mask=mask)


def rmsnorm_naive(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and w.dim() == 1 and w.shape[0] == x.shape[1]
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    # Hardcoded BLOCK_SIZE — picked by "looks reasonable", not measured.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    rmsnorm_naive_kernel[(n_rows,)](out, x, w, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


def reference(x, w, eps=1e-6):
    """Reference RMSNorm in pure PyTorch."""
    x_fp32 = x.to(torch.float32)
    rms = torch.sqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + eps)
    return ((x_fp32 / rms) * w.to(torch.float32)).to(x.dtype)


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    n_rows, n_cols = 4096, 4096
    x = torch.randn(n_rows, n_cols, device=device, dtype=dtype) * 0.1
    w = torch.randn(n_cols, device=device, dtype=dtype) * 0.5 + 1.0

    out_triton = rmsnorm_naive(x, w)
    out_ref = reference(x, w)
    diff = (out_triton - out_ref).abs().max().item()
    print(f"correctness: max abs diff = {diff:.2e}")
    assert diff < 1e-2, "too large — debug"

    ms = triton.testing.do_bench(lambda: rmsnorm_naive(x, w))
    bytes_io = (2 * n_rows * n_cols + n_cols) * x.element_size() + n_rows * n_cols * x.element_size()
    # The factor "2 * n_rows * n_cols" reflects the two loads of x in this naive kernel.
    # The "real" minimum is `(n_rows * n_cols + n_cols) + n_rows * n_cols` = ~3 N_rows*N_cols
    # but our kernel does double-load so the effective bytes-moved is 4*N_rows*N_cols + n_cols.
    gbps = bytes_io / (ms * 1e-3) / 1e9

    print(f"\nnaive RMSNorm  n_rows={n_rows} n_cols={n_cols} {dtype}")
    print(f"  time: {ms:.3f} ms")
    print(f"  effective bytes moved: {bytes_io/1e9:.2f} GB (counting double-load of x)")
    print(f"  GB/s (bandwidth used): {gbps:.1f}")
    print(f"  Compare to your GPU's peak HBM bandwidth — this should be ~10-15% of peak.")


if __name__ == "__main__":
    main()
