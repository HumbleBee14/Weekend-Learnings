"""
A fused kernel that computes `softmax(x) * y` in one pass.

Why this matters: in eager PyTorch, `softmax(x) * y` runs softmax (which round-trips
intermediate values to HBM), then a separate elementwise multiply (another round-trip).
With Triton, you write one kernel that reads x and y once, computes softmax in registers,
multiplies by y, writes the output once.

This is the simplest example of why Triton (and FlashAttention, in Topic 6) wins:
HBM round-trip elimination.

Run:
    pip install triton torch
    python fused_softmax_times_y.py
"""

import time
import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_y_kernel(
    x_ptr, y_ptr, out_ptr,
    n_cols,
    x_row_stride, y_row_stride, o_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. Computes softmax(x[row]) * y[row] in registers."""
    row_idx = tl.program_id(0)

    x_row = x_ptr + row_idx * x_row_stride
    y_row = y_ptr + row_idx * y_row_stride
    o_row = out_ptr + row_idx * o_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load x[row] tile. -inf for masked positions so they don't affect the max.
    x = tl.load(x_row + col_offsets, mask=mask, other=-float("inf"))

    # Numerically stable softmax: subtract max, exp, normalize.
    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)
    denominator = tl.sum(numerator, axis=0)
    sm = numerator / denominator

    # Multiply by y, all in registers. No intermediate write.
    y = tl.load(y_row + col_offsets, mask=mask, other=0.0)
    out = sm * y

    tl.store(o_row + col_offsets, out, mask=mask)


def fused_softmax_y(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape and x.dim() == 2
    assert x.is_cuda and y.is_cuda
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    # One program per row; BLOCK_SIZE >= n_cols rounded up to next pow2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    fused_softmax_y_kernel[(n_rows,)](
        x, y, out,
        n_cols,
        x.stride(0), y.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")

    n_rows, n_cols = 1024, 4096
    x = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float32)
    y = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float32)

    out_triton = fused_softmax_y(x, y)
    out_torch = torch.softmax(x, dim=-1) * y
    print(f"correctness: {torch.allclose(out_triton, out_torch, atol=1e-5)}")

    def bench(fn, name):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 10
        bytes_moved = 3 * n_rows * n_cols * 4   # 2 reads + 1 write
        bw = bytes_moved / 1e9 / (ms / 1000)
        print(f"  {name:<22} {ms:.3f} ms,  {bw:.0f} GB/s")

    print(f"\n{n_rows} × {n_cols} (fp32):")
    bench(lambda: torch.softmax(x, dim=-1) * y, "torch (unfused)")
    bench(lambda: fused_softmax_y(x, y),         "triton (fused)")

    print("\nFused = one HBM read of x + one of y + one write of output.")
    print("Unfused = read x, write softmax(x), read softmax(x) and y, write product.")
    print("Roughly 1.5–2× bandwidth saving from elimination of the intermediate write/read.")


if __name__ == "__main__":
    main()
