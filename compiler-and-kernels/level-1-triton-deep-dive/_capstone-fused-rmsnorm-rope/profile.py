"""
Verify the bandwidth claim with the proton profiler.

The numbers to check:
  1. dram__bytes_read   ≈ B*S*H*2 (input) + small (weight + cos/sin amortized for persistent)
  2. dram__bytes_write  ≈ B*S*H*2 (output)
  3. dram__throughput pct > 80% for the persistent variant
  4. sm__warps_active pct > 70%

If you see 2× the expected reads, you're loading something twice. The most common
cause in this fused kernel is the partner-element load via cols ^ 1 — that's an
extra HBM transaction if the L1 doesn't catch it. Some kernels avoid it via a
smarter register-side gather; see Liger-Kernel's approach for one option.
"""

import torch, os
from reference import build_rope_tables
from fused_rmsnorm_rope import fused_persistent


def main():
    try:
        import triton.profiler as proton
    except ImportError:
        print("triton.profiler not available. Try: pip install --upgrade triton")
        return

    torch.manual_seed(0)
    B, S, H = 32, 2048, 4096
    x = torch.randn(B, S, H, device="cuda", dtype=torch.float16) * 0.1
    w = (torch.randn(H, device="cuda", dtype=torch.float16) * 0.5 + 1.0)
    cos, sin = build_rope_tables(8192, H, device="cuda")
    pos = torch.arange(S, device="cuda", dtype=torch.int64).expand(B, S).contiguous()

    # Warm up to skip autotune in the trace
    for _ in range(5):
        _ = fused_persistent(x, w, cos, sin, pos)

    trace_path = os.path.join(os.path.dirname(__file__), "capstone.proton.trace")
    proton.start(trace_path)
    with proton.scope("fused_rmsnorm_rope_persistent"):
        for _ in range(20):
            _ = fused_persistent(x, w, cos, sin, pos)
    proton.finalize()

    print(f"Trace at {trace_path}")
    print(f"View with: proton-viewer {trace_path}")
    print()
    print("Expected DRAM bytes per iteration (approximate):")
    print(f"  read input:   {B * S * H * 2 / 1e6:.1f} MB")
    print(f"  read weight:  {H * 2 / 1024:.1f} KB (amortized once per program)")
    print(f"  read cos+sin: ~{S * H * 2 / 1e6:.1f} MB upper bound (gathered, may be less)")
    print(f"  write output: {B * S * H * 2 / 1e6:.1f} MB")
    print()
    print("If the trace shows 2x the expected input bytes, find the duplicated load.")


if __name__ == "__main__":
    main()
