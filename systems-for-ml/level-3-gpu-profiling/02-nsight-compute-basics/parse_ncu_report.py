"""
Read an .ncu-rep file in Python and pull out the Speed-of-Light numbers.

This is the pattern you'd use in CI to detect kernel performance regressions.

Setup:
    pip install ncu-report     # NVIDIA-provided package

Usage:
    # First, generate a report file:
    ncu --set basic -k regex:fast_kernel -c 1 -o report.ncu-rep python profile_a_kernel.py

    # Then parse it:
    python parse_ncu_report.py report.ncu-rep
"""

import sys

try:
    from ncu_report import load_report
except ImportError:
    raise SystemExit("pip install ncu-report")


def main(path: str):
    report = load_report(path)
    print(f"Report: {path}")
    print(f"Kernels captured: {len(report)}\n")

    for k in report:
        # Each `k` is one kernel invocation. Common metrics:
        try:
            compute_sol = k["sm__throughput.avg.pct_of_peak_sustained_elapsed"].value()
        except KeyError:
            compute_sol = None
        try:
            memory_sol = k["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"].value()
        except KeyError:
            memory_sol = None
        try:
            duration_us = k["gpu__time_duration.sum"].value() / 1000  # ns → µs
        except KeyError:
            duration_us = None
        try:
            occupancy = k["sm__warps_active.avg.pct_of_peak_sustained_active"].value()
        except KeyError:
            occupancy = None

        print(f"  {k.name()}")
        if duration_us is not None:
            print(f"    duration:     {duration_us:.1f} µs")
        if compute_sol is not None:
            print(f"    compute SOL:  {compute_sol:.1f}%")
        if memory_sol is not None:
            print(f"    memory SOL:   {memory_sol:.1f}%")
        if occupancy is not None:
            print(f"    occupancy:    {occupancy:.1f}%")

        # Diagnose
        if compute_sol is not None and memory_sol is not None:
            if compute_sol > 60 and memory_sol < 50:
                print("    → COMPUTE-BOUND")
            elif memory_sol > 60 and compute_sol < 50:
                print("    → MEMORY-BOUND (HBM bandwidth limited)")
            elif compute_sol > 50 and memory_sol > 50:
                print("    → BALANCED (well-utilized)")
            else:
                print("    → LATENCY-BOUND (low utilization on both axes)")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_ncu_report.py <path-to-.ncu-rep>")
        sys.exit(1)
    main(sys.argv[1])
