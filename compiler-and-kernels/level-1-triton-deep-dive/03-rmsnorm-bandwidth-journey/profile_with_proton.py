"""
Profile your fastest RMSNorm kernel with triton.proton.

What this script verifies — the things you should confirm before declaring done:

  1. dram__bytes_read   ≈ (n_rows * n_cols + n_cols) * dtype_size
     (One read of the input, one read of the weight, amortized to ~once if persistent.)
  2. dram__bytes_write  ≈ n_rows * n_cols * dtype_size
     (One write of the output.)
  3. sm__warps_active.avg.pct_of_peak_sustained_active > 70%
     (SMs are busy.)
  4. dram__throughput.avg.pct_of_peak_sustained_elapsed > 80%
     (We're using the memory bus.)

If any number is off — especially if read/write bytes are 2x what you expect — you have a hidden re-load
or your kernel never actually executed (cache miss, grid 0, etc.). Always sanity-check by also running
the correctness assertion.

Requires: Triton 3.x with proton (`triton.profiler` module). Most Triton 3.4+ installs ship it.
If `import triton.profiler` fails, install via `pip install --upgrade triton`.
"""

import os, importlib.util, torch, triton

HERE = os.path.dirname(os.path.abspath(__file__))
v05 = importlib.util.spec_from_file_location("v05", os.path.join(HERE, "05_persistent.py"))
v05_mod = importlib.util.module_from_spec(v05)
v05.loader.exec_module(v05_mod)


def main():
    try:
        import triton.profiler as proton
    except ImportError:
        print("triton.profiler not found. Upgrade triton: pip install --upgrade triton")
        return

    torch.manual_seed(0)
    n_rows, n_cols = 4096, 4096
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float16) * 0.1
    w = torch.randn(n_cols, device="cuda", dtype=torch.float16) * 0.5 + 1.0

    # Warm up to bypass autotune in the profiled region.
    for _ in range(5):
        _ = v05_mod.rmsnorm_persistent(x, w)

    trace_path = os.path.join(HERE, "rmsnorm_v5.proton.trace")
    proton.start(trace_path)
    with proton.scope("rmsnorm_v5"):
        for _ in range(50):
            _ = v05_mod.rmsnorm_persistent(x, w)
    proton.finalize()

    print(f"Trace written to {trace_path}")
    print()
    print("To view:")
    print(f"  proton-viewer {trace_path}")
    print()
    print("Look for these metrics in the trace:")
    print("  - dram__bytes_read         ≈ {:.2f} MB per iteration".format(
        (n_rows * n_cols + n_cols) * 2 / 1024**2))
    print("  - dram__bytes_write        ≈ {:.2f} MB per iteration".format(
        n_rows * n_cols * 2 / 1024**2))
    print("  - dram__throughput pct     should be > 80%")
    print("  - sm__warps_active pct     should be > 70%")
    print()
    print("If any of these are off, your fast kernel may be cheating (no-op) or wasteful (double-load).")
    print("Cross-check with the correctness assertion built into 05_persistent.py.")


if __name__ == "__main__":
    main()
