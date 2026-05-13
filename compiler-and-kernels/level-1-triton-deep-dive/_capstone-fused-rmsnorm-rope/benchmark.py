"""
Capstone benchmark: your three fused variants vs eager, vs torch.compile,
vs Liger-Kernel.

Produces the comparison table you put in report.md.

Install liger-kernel for the production reference:
  pip install liger-kernel
"""

import torch, triton
from reference import rmsnorm_rope_reference, build_rope_tables
from fused_rmsnorm_rope import fused_naive, fused_autotuned, fused_persistent


def try_liger_combined(x, w, cos, sin, pos):
    """
    Liger doesn't ship a single fused RMSNorm+RoPE op — it ships them separately.
    For a fair comparison we run both Liger ops back-to-back as the "production"
    baseline that doesn't fuse them. (This is what most LLM training stacks today
    actually run.) If you want a single-fused production reference, Mistral's
    `mistral-common` and some inference engines do it, but they aren't pip-installable
    as cleanly.
    """
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        from liger_kernel.ops.rope import LigerRopeFunction
    except ImportError:
        return None

    H = x.shape[-1]
    flat = x.reshape(-1, H)
    pos_flat = pos.reshape(-1)

    def run():
        # Liger RMSNorm forward: takes [N, H], weight, eps, casting_mode
        normed = LigerRMSNormFunction.apply(flat, w, 1e-6, 0, "llama", True)
        # Liger RoPE: takes (q, cos, sin) — uses interleaved form
        # We apply it as if `normed` were the q-projection (same shape).
        normed3 = normed.reshape(x.shape).unsqueeze(2)  # add head dim of 1
        cos_3 = cos[pos_flat].reshape(*x.shape[:-1], H // 2)
        sin_3 = sin[pos_flat].reshape(*x.shape[:-1], H // 2)
        out_3, _ = LigerRopeFunction.apply(normed3, normed3, cos_3, sin_3)
        return out_3.squeeze(2)

    return run


def eager_reference(x, w, cos, sin, pos):
    return rmsnorm_rope_reference(x, w, cos, sin, pos)


@torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)
def compiled_reference(x, w, cos, sin, pos):
    return rmsnorm_rope_reference(x, w, cos, sin, pos)


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # Production-shape: 32 batches of 2048 tokens, hidden=4096 (Llama-7B-ish).
    B, S, H = 32, 2048, 4096
    max_seqlen = 8192

    x = torch.randn(B, S, H, device=device, dtype=dtype) * 0.1
    w = (torch.randn(H, device=device, dtype=dtype) * 0.5 + 1.0)
    cos, sin = build_rope_tables(max_seqlen, H, device=device)
    pos = torch.arange(S, device=device, dtype=torch.int64).expand(B, S).contiguous()

    # Warmup torch.compile outside timed region
    _ = compiled_reference(x, w, cos, sin, pos)

    # Theoretical-minimum bytes moved (single-fused, persistent):
    #   read x:        B*S*H*2
    #   read w:        H*2 (amortized once per program; with persistent grid, ~ num_SMs * H * 2)
    #   read cos+sin:  ~ S * H * 2 worth (gathered, hard to give exact; use upper bound)
    #   write out:     B*S*H*2
    # We use a simple "input + output" lower bound: 2 * B*S*H*2.
    min_bytes = 2 * B * S * H * 2

    try:
        name = torch.cuda.get_device_name(0).lower()
        peak_map = {"t4": 320, "a100": 1555, "h100": 3350, "h200": 4800, "b200": 8000,
                    "rtx 4090": 1008, "rtx 5090": 1792, "mi300x": 5300}
        peak_bw = next((v for k, v in peak_map.items() if k in name), None)
    except Exception:
        peak_bw = None

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: B={B} S={S} H={H} {dtype}  (min input+output traffic = {min_bytes/1e9:.2f} GB)")
    if peak_bw:
        print(f"Approx HBM peak: {peak_bw} GB/s")
    print()

    impls = [
        ("eager pytorch",       lambda: eager_reference(x, w, cos, sin, pos)),
        ("torch.compile",       lambda: compiled_reference(x, w, cos, sin, pos)),
        ("yours: naive",        lambda: fused_naive(x, w, cos, sin, pos)),
        ("yours: autotuned",    lambda: fused_autotuned(x, w, cos, sin, pos)),
        ("yours: persistent",   lambda: fused_persistent(x, w, cos, sin, pos)),
    ]

    liger_run = try_liger_combined(x, w, cos, sin, pos)
    if liger_run is not None:
        # Warmup
        for _ in range(3):
            _ = liger_run()
        impls.append(("liger (separate)", liger_run))
    else:
        print("(liger-kernel not installed; skipping. pip install liger-kernel)\n")

    print(f"{'impl':<22}{'ms':>10}{'GB/s':>12}{'% peak':>10}{'vs eager':>10}")
    print("-" * 64)

    # Get eager time first so we can compute speedup column
    ms_eager = triton.testing.do_bench(impls[0][1])

    for name, fn in impls:
        ms = triton.testing.do_bench(fn)
        gbps = min_bytes / (ms * 1e-3) / 1e9
        pct = f"{gbps / peak_bw * 100:.0f}%" if peak_bw else "—"
        speedup = f"{ms_eager / ms:.2f}x"
        print(f"{name:<22}{ms:>10.3f}{gbps:>12.1f}{pct:>10}{speedup:>10}")

    print()
    print("If your persistent version is within 5% (% peak) of Liger, you matched production.")
    print("If you beat Liger: re-check measurement (warmup, dtype, same eps, correctness).")
    print()
    print("Write up these numbers in report.md with your interpretation.")


if __name__ == "__main__":
    main()
