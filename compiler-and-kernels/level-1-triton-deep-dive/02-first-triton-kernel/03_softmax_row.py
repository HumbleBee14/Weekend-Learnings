"""
03 — Row-softmax in Triton.

For each row of a 2D tensor, compute softmax across the columns.
This is the kernel shape behind the last step of attention (softmax over the K dimension).

The kernel uses one program per row. Inside each program, the full row fits
in registers (we assume N_COLS <= BLOCK_SIZE), so we can use the simple
two-step subtract-max-then-normalize formulation:

    m = max(x)
    s = sum(exp(x - m))
    out = exp(x - m) / s

The "online softmax" version (running max and sum updated in a single scan)
is the building block of FlashAttention. We meet the algebra here on a small
example and re-use it for real in Level 3.

Online softmax derivation, worked example.
  Suppose row x = [1.0, 4.0, 2.0].
  Two-pass:
    m = 4.0
    s = exp(1-4) + exp(4-4) + exp(2-4) = exp(-3) + 1 + exp(-2)
      = 0.0498 + 1.0 + 0.1353 = 1.1851
    out = [exp(-3)/1.1851, 1/1.1851, exp(-2)/1.1851] = [0.0420, 0.8438, 0.1142]

  Online (one pass, update m and s as you scan):
    Step 0: m=-inf, s=0
    See x=1.0:  m_new = max(-inf, 1.0) = 1.0;  s = s*exp(-inf-1.0) + exp(1.0-1.0) = 0+1 = 1.0
    See x=4.0:  m_new = max(1.0, 4.0) = 4.0;  s = s*exp(1.0-4.0) + exp(4.0-4.0)
                                                 = 1.0*exp(-3) + 1.0 = 0.0498 + 1.0 = 1.0498
    See x=2.0:  m_new = max(4.0, 2.0) = 4.0;  s = s*exp(4.0-4.0) + exp(2.0-4.0)
                                                 = 1.0498*1.0 + exp(-2) = 1.0498 + 0.1353 = 1.1851

  Same s as the two-pass. Same m. Then out = exp(x - m) / s in a second pass.

  The key trick is the line `s = s*exp(m_old - m_new) + exp(x - m_new)`. When a
  new larger max is seen, all old contributions to s were "too big" relative to
  the new max and need to be scaled down by exp(m_old - m_new). New contributions
  are scaled fresh relative to the new max.

  This single-pass property is what lets FlashAttention process Q @ K^T tile by
  tile without materializing the full attention matrix — we accumulate the
  softmax stats as we sweep K, rescaling whenever we see a larger element.

What you should observe when you run this:
  - Correctness: max diff vs torch.softmax is small (FP precision, ~1e-6).
  - Throughput on a (batch=2048, cols=4096) tensor: bound by HBM. T4 ~200 GB/s,
    A100 ~1500 GB/s, H100 ~2500 GB/s.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, x_ptr,
    x_row_stride, out_row_stride,   # stride between rows in elements (usually = n_cols for contiguous)
    n_cols,                          # int — number of columns per row
    BLOCK_SIZE: tl.constexpr,        # tile width; must be >= n_cols (we assume row fits in one tile)
):
    # One program per row.
    row_idx = tl.program_id(axis=0)

    # Pointers to the start of this row.
    x_row = x_ptr + row_idx * x_row_stride
    out_row = out_ptr + row_idx * out_row_stride

    # Column indices for this tile.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row. Mask out-of-bounds with -inf so they don't affect max/sum.
    x = tl.load(x_row + col_offsets, mask=mask, other=-float("inf"))

    # Subtract the row max for numerical stability, then normalize.
    # tl.max here is a reduction across the BLOCK_SIZE-wide vector inside one program.
    m = tl.max(x, axis=0)
    x_shifted = x - m
    numerator = tl.exp(x_shifted)
    denom = tl.sum(numerator, axis=0)
    out = numerator / denom

    tl.store(out_row + col_offsets, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-softmax on a 2D tensor [n_rows, n_cols]."""
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape

    # Triton requires BLOCK_SIZE to be a power of 2.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # If your row is huge (>16K), the row won't fit in registers and you'd need
    # the online-softmax pattern across multiple tiles per row. We assume not here.
    assert BLOCK_SIZE <= 16384, "Row too wide for the simple kernel; use online softmax."

    out = torch.empty_like(x)
    grid = (n_rows,)  # one program per row

    softmax_kernel[grid](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"

    # Correctness across a few shapes including non-power-of-2 column counts
    for shape in [(8, 16), (32, 4097), (2048, 1024), (1, 4096)]:
        x = torch.randn(shape, device=device, dtype=torch.float32)
        out_triton = softmax(x)
        out_torch = torch.softmax(x, dim=1)
        max_diff = (out_triton - out_torch).abs().max().item()
        assert max_diff < 1e-5, f"shape={shape}: diff={max_diff}"
        print(f"shape={shape}  max_diff={max_diff:.2e}  OK")

    # Benchmark
    n_rows, n_cols = 2048, 4096
    x = torch.randn(n_rows, n_cols, device=device, dtype=torch.float32)

    ms_triton = triton.testing.do_bench(lambda: softmax(x))
    ms_torch = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))

    # Bytes moved: load N*M floats, store N*M floats.
    bytes_moved = 2 * n_rows * n_cols * 4
    gbps_triton = bytes_moved / (ms_triton * 1e-3) / 1e9
    gbps_torch = bytes_moved / (ms_torch * 1e-3) / 1e9

    print()
    print(f"shape = ({n_rows}, {n_cols})")
    print(f"  triton: {ms_triton:.3f} ms   {gbps_triton:7.1f} GB/s")
    print(f"  torch : {ms_torch:.3f} ms   {gbps_torch:7.1f} GB/s")
    print()
    print("Both should approach HBM peak. torch.softmax is hand-tuned cuBLAS/cuDNN;")
    print("matching it with a few lines of Triton is the headline of this sub-module.")


if __name__ == "__main__":
    main()
