"""
Step 05 — Persistent kernel: launch exactly num_SMs programs, each one
processes multiple rows in a loop.

Two wins:
  1. No scheduler thrash: 132 programs once instead of ~500 waves of scheduling.
  2. Weight reuse from L2: each SM loads the weight vector from HBM once
     (for its first row), then hits it from L2 for the rest of its rows.

Expected: 80-90% of peak HBM bandwidth. Effectively the ceiling for this
operator class. The remaining gap to absolute peak comes from warp specialization
(sub-module 05) which we don't apply here because the algorithm is too simple
to benefit much from it.
"""

import torch
import triton
import triton.language as tl


def _prune_configs(configs, named_args, **kwargs):
    n_cols = named_args["n_cols"]
    target_block = triton.next_power_of_2(n_cols)
    pruned = []
    for cfg in configs:
        bs = cfg.kwargs["BLOCK_SIZE"]
        if bs < target_block or bs > 2 * target_block:
            continue
        if bs % (cfg.num_warps * 32) != 0:
            continue
        pruned.append(cfg)
    return pruned


_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
    for bs in [1024, 2048, 4096, 8192]
    for nw in [2, 4, 8, 16]
    for ns in [2, 3, 4]
]


@triton.autotune(configs=_CONFIGS, key=["n_cols"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def rmsnorm_persistent_kernel(
    out_ptr, x_ptr, w_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Each program processes rows [pid, pid + num_pids, pid + 2*num_pids, ...].
    # The weight vector is loaded once at the top — every subsequent iteration
    # reuses the same register tile. Subsequent rows hit weights from L2 (the SM
    # already loaded them into the cache hierarchy).
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for row in range(pid, n_rows, num_pids):
        x_row = x_ptr + row * n_cols
        out_row = out_ptr + row * n_cols

        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
        out = (x * inv_rms) * w
        tl.store(out_row + cols, out.to(tl.float16), mask=mask)


def rmsnorm_persistent(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    # Grid size = number of SMs. We cap so we never launch more programs than rows.
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = (min(num_sms, n_rows),)
    rmsnorm_persistent_kernel[grid](out, x, w, n_rows, n_cols, eps)
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

    print("Autotuning (first call)...")
    out_t = rmsnorm_persistent(x, w)
    out_r = reference(x, w)
    print(f"correctness: max diff = {(out_t - out_r).abs().max().item():.2e}")
    print(f"best config: {rmsnorm_persistent_kernel.best_config}")

    ms = triton.testing.do_bench(lambda: rmsnorm_persistent(x, w))
    bytes_io = (n_rows * n_cols + n_cols + n_rows * n_cols) * 2
    gbps = bytes_io / (ms * 1e-3) / 1e9

    n_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    print(f"\nGPU has {n_sms} SMs. Persistent grid launched {min(n_sms, n_rows)} programs.")
    print(f"persistent RMSNorm: {ms:.3f} ms, {gbps:.1f} GB/s — should be 1.1-1.3x step 04.")
    print("\nThis is roughly the ceiling for this operator on this GPU without warp specialization.")


if __name__ == "__main__":
    main()
