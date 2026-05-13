"""
Step 03 — Single pass: load x ONCE, keep it in registers, use it for both
the reduction and the elementwise normalize.

This is the biggest single bandwidth win in the journey: we halve the
input traffic by not reloading the row for the second pass.

Expected: 50–60% of peak HBM bandwidth.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_1pass_kernel(
    out_ptr, x_ptr, w_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    # SINGLE PASS. We load x once into registers (`x` is now a register tile),
    # use it for the reduction, then use the same registers for the normalize.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    rms = tl.sqrt(sum_sq / n_cols + eps)
    inv_rms = 1.0 / rms

    # `x` is still in registers — no reload needed.
    out = (x * inv_rms) * w
    tl.store(out_row + cols, out.to(tl.float16), mask=mask)


def rmsnorm_1pass(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float16
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8 if BLOCK_SIZE >= 2048 else 4
    rmsnorm_1pass_kernel[(n_rows,)](out, x, w, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return out


def reference(x, w, eps=1e-6):
    x_fp32 = x.to(torch.float32)
    rms = torch.sqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + eps)
    return ((x_fp32 / rms) * w.to(torch.float32)).to(x.dtype)


def main():
    torch.manual_seed(0)
    n_rows, n_cols = 4096, 4096
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn(n_cols, device="cuda", dtype=torch.float16) * 0.5 + 1.0

    out_t = rmsnorm_1pass(x, w)
    out_r = reference(x, w)
    print(f"correctness: max diff = {(out_t - out_r).abs().max().item():.2e}")

    ms = triton.testing.do_bench(lambda: rmsnorm_1pass(x, w))
    # Single-pass bytes-moved: load x once (N*M*2), load w once (M*2), store out once (N*M*2).
    bytes_io = (n_rows * n_cols + n_cols + n_rows * n_cols) * 2
    gbps = bytes_io / (ms * 1e-3) / 1e9
    print(f"\nsingle-pass RMSNorm: {ms:.3f} ms, {gbps:.1f} GB/s — should be 1.7-2x step 02.")
    print("(Half the input bytes vs the two-pass version.)")


if __name__ == "__main__":
    main()
