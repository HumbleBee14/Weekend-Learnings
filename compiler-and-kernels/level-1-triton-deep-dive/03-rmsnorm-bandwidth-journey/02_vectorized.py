"""
Step 02 — Vectorized: bigger tiles, more warps, cache hints.

Same algorithm as step 01 (two passes), but tile size is the full row and
num_warps is sized so each warp gets a reasonable chunk of the row.
Cache hint on the second load reduces L1 pollution.

Expected: 25–35% of peak HBM bandwidth. Two passes are still the dominant cost,
but at least we're saturating the memory bus on each pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_vec_kernel(
    out_ptr, x_ptr, w_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    # PASS 1: regular load, used for the reduction.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    rms = tl.sqrt(sum_sq / n_cols + eps)

    # PASS 2: cache-streaming load (`.cs`) — we use this data once and won't reuse.
    # Bypasses L1 to avoid polluting it. On Hopper this hint maps to a streaming load;
    # on T4 it's advisory and may be ignored.
    x_again = tl.load(x_row + cols, mask=mask, other=0.0, cache_modifier=".cs").to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x_again / rms) * w
    tl.store(out_row + cols, out.to(tl.float16), mask=mask)


def rmsnorm_vec(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float16
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # num_warps tuned for 4096-wide tile: 8 warps × 32 lanes = 256 active lanes,
    # each handling 16 elements. Good occupancy on H100, T4, and A100.
    num_warps = 8 if BLOCK_SIZE >= 2048 else 4
    rmsnorm_vec_kernel[(n_rows,)](out, x, w, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
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

    out_t = rmsnorm_vec(x, w)
    out_r = reference(x, w)
    print(f"correctness: max diff = {(out_t - out_r).abs().max().item():.2e}")

    ms = triton.testing.do_bench(lambda: rmsnorm_vec(x, w))
    # Same double-load as before, so bytes-moved is still ~4*N*M*2 (fp16).
    bytes_io = (2 * n_rows * n_cols + n_cols + n_rows * n_cols) * 2
    gbps = bytes_io / (ms * 1e-3) / 1e9
    print(f"\nvectorized RMSNorm: {ms:.3f} ms, {gbps:.1f} GB/s — should be 2-3x step 01.")


if __name__ == "__main__":
    main()
