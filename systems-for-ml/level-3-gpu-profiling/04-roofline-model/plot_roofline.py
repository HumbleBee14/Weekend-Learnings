"""
Plot a roofline chart for a few common LLM kernels on a chosen GPU.

Run:
    pip install torch matplotlib
    python plot_roofline.py

Outputs roofline.png with:
  - Slanted line (memory bandwidth ceiling)
  - Flat line (compute ceiling)
  - Plotted points for: GEMM (large), GEMM (decode shape), attention, RMSNorm
"""

import matplotlib.pyplot as plt
import numpy as np


# Pick your GPU. Edit these to match what you're running on.
# (Use ERT or just the marketed peak; a real curve sits 70-85% lower.)
GPU_NAME = "H100 SXM"
PEAK_TFLOPS = 989                 # BF16 dense
PEAK_HBM_BW_TBPS = 3.35           # HBM3
PEAK_HBM_BW_BYTES_PER_S = PEAK_HBM_BW_TBPS * 1e12
PEAK_FLOPS = PEAK_TFLOPS * 1e12

ridge_ai = PEAK_FLOPS / PEAK_HBM_BW_BYTES_PER_S
print(f"{GPU_NAME}: peak {PEAK_TFLOPS} TFLOPS, HBM {PEAK_HBM_BW_TBPS} TB/s, ridge AI = {ridge_ai:.0f} FLOP/byte")


def estimate_kernel(name: str, flops: int, bytes_moved: int, achieved_tflops: float):
    """Return (name, AI, achieved_tflops) for plotting."""
    ai = flops / bytes_moved
    return name, ai, achieved_tflops


def main():
    # Estimate AI and (rough) achieved performance for a few kernels.
    # Achieved values are approximate — your real kernel performance depends on tuning.
    kernels = [
        # GEMM 4096³ in BF16 — compute-bound, well-tuned
        # FLOPs: 2 * 4096^3 ≈ 1.37e11; Bytes: 2 * 3 * 4096^2 ≈ 1.0e8
        estimate_kernel("GEMM 4096³ BF16", 2 * 4096**3, 2 * 3 * 4096**2, achieved_tflops=750),

        # GEMM decode shape: M=1, N=K=4096
        # FLOPs: 2 * 1 * 4096^2 ≈ 3.4e7; Bytes: ~2 * (4096^2) — read entire weight matrix
        estimate_kernel("GEMM decode (M=1)", 2 * 1 * 4096 * 4096, 2 * 4096 * 4096, achieved_tflops=2.0),

        # FlashAttention forward, B=1, H=8, N=4096, D=128 (rough numbers)
        # FLOPs: 4 * B * H * N² * D ≈ 4 * 1 * 8 * 4096² * 128 ≈ 6.9e10
        # Bytes: 2 * 4 * B * H * N * D ≈ 3.4e7  (Q, K, V, O — no N×N intermediate)
        estimate_kernel("FlashAttention N=4096", 4 * 8 * 4096**2 * 128, 2 * 4 * 8 * 4096 * 128, achieved_tflops=600),

        # RMSNorm on a 4096-element vector, fp16 — pure memory bound
        # FLOPs: ~3 * 4096; Bytes: 2 * 2 * 4096 (read + write)
        estimate_kernel("RMSNorm (per token)", 3 * 4096, 2 * 2 * 4096, achieved_tflops=0.05),
    ]

    # Plot roofline
    ai_range = np.logspace(0, 4, 200)  # 1 to 10000 FLOP/byte
    perf_memory = ai_range * PEAK_HBM_BW_BYTES_PER_S / 1e12  # TFLOPS
    perf_compute = np.full_like(ai_range, PEAK_TFLOPS)
    perf_ceiling = np.minimum(perf_memory, perf_compute)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai_range, perf_ceiling, "k-", linewidth=2, label="Roofline ceiling")
    ax.loglog(ai_range, perf_memory, "b--", alpha=0.4, label=f"Memory ceiling ({PEAK_HBM_BW_TBPS} TB/s)")
    ax.loglog(ai_range, perf_compute, "r--", alpha=0.4, label=f"Compute ceiling ({PEAK_TFLOPS} TFLOPS)")

    # Mark the ridge point
    ax.axvline(ridge_ai, color="gray", linestyle=":", alpha=0.5)
    ax.text(ridge_ai * 1.1, 1.5, f"ridge AI = {ridge_ai:.0f}", color="gray")

    # Plot the kernels
    for name, ai, perf in kernels:
        ax.scatter(ai, perf, s=100, zorder=5)
        ax.annotate(f"  {name}\n  AI={ai:.1f}, perf={perf:.1f} TFLOPS",
                    (ai, perf), fontsize=9, va="center")

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (TFLOPS, BF16)")
    ax.set_title(f"Roofline plot — {GPU_NAME}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(0.01, PEAK_TFLOPS * 2)

    fig.tight_layout()
    fig.savefig("roofline.png", dpi=120)
    print("Wrote roofline.png")

    # Also print analysis
    print(f"\nKernel analysis on {GPU_NAME} (ridge AI = {ridge_ai:.0f}):")
    for name, ai, perf in kernels:
        regime = "compute-bound" if ai > ridge_ai else "memory-bound"
        if ai > ridge_ai:
            ceiling = PEAK_TFLOPS
        else:
            ceiling = ai * PEAK_HBM_BW_TBPS  # TFLOPS
        utilization = perf / ceiling * 100
        print(f"  {name}: AI={ai:.1f}, achieved={perf:.1f} TFLOPS, ceiling={ceiling:.1f}, "
              f"util={utilization:.0f}%  ({regime})")


if __name__ == "__main__":
    main()
