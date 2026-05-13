"""
Step 04 — Autotuned with early_config_prune.

Same single-pass algorithm, but the compiler is allowed to pick BLOCK_SIZE,
num_warps, and num_stages from a search space we filter intelligently.

The discipline: we don't sweep nonsense configs. We prune away ones that:
  - Would need multiple tiles per row (BLOCK_SIZE < n_cols) — incompatible with
    the one-tile-per-row shape of this kernel
  - Would waste SRAM (BLOCK_SIZE much larger than next_power_of_2(n_cols))
  - Have num_warps that doesn't fit the tile size cleanly

Expected: 70–80% of peak HBM bandwidth. The compiler often picks something
non-obvious you wouldn't have guessed (e.g., num_stages=3 instead of 2).
"""

import torch
import triton
import triton.language as tl


def _prune_configs(configs, named_args, **kwargs):
    """Filter configs that don't fit the runtime shape."""
    n_cols = named_args["n_cols"]
    target_block = triton.next_power_of_2(n_cols)
    pruned = []
    for cfg in configs:
        bs = cfg.kwargs["BLOCK_SIZE"]
        # Must be >= n_cols (one tile per row) and not absurdly large.
        if bs < target_block:
            continue
        if bs > 2 * target_block:
            continue
        # num_warps * 32 should divide BLOCK_SIZE for clean lane allocation.
        if bs % (cfg.num_warps * 32) != 0:
            continue
        pruned.append(cfg)
    return pruned


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
    for bs in [1024, 2048, 4096, 8192]
    for nw in [2, 4, 8, 16]
    for ns in [2, 3, 4]
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_cols"],  # re-tune when n_cols changes
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def rmsnorm_auto_kernel(
    out_ptr, x_ptr, w_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)

    out = (x * inv_rms) * w
    tl.store(out_row + cols, out.to(tl.float16), mask=mask)


def rmsnorm_auto(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    rmsnorm_auto_kernel[(n_rows,)](out, x, w, n_cols, eps)
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

    # First call runs the autotune. Note the latency.
    print("Autotuning (first call)...")
    out_t = rmsnorm_auto(x, w)
    out_r = reference(x, w)
    print(f"correctness: max diff = {(out_t - out_r).abs().max().item():.2e}")

    # Inspect the winning config.
    print("\nBest config picked:")
    print(f"  {rmsnorm_auto_kernel.best_config}")

    ms = triton.testing.do_bench(lambda: rmsnorm_auto(x, w))
    bytes_io = (n_rows * n_cols + n_cols + n_rows * n_cols) * 2
    gbps = bytes_io / (ms * 1e-3) / 1e9
    print(f"\nautotuned RMSNorm: {ms:.3f} ms, {gbps:.1f} GB/s — should be 1.3-1.5x step 03.")


if __name__ == "__main__":
    main()
