# 02 — Nsight Compute Basics

## Files

- `CONCEPTS.md` — Speed of Light, occupancy myths, warp stall reasons, the workflow, modern CI integration
- `profile_a_kernel.py` — two Triton kernels (fast and slow) doing the same operation with different memory access patterns
- `parse_ncu_report.py` — example of reading `.ncu-rep` files in Python (the CI regression-testing pattern)

## Quickstart

```bash
pip install torch triton ncu-report

# Run the script once to JIT-compile both kernels
python profile_a_kernel.py

# Profile each with ncu basic (fast)
ncu --set basic -k regex:fast_kernel -c 1 -o fast.ncu-rep python profile_a_kernel.py
ncu --set basic -k regex:slow_kernel -c 1 -o slow.ncu-rep python profile_a_kernel.py

# Parse the reports in Python
python parse_ncu_report.py fast.ncu-rep
python parse_ncu_report.py slow.ncu-rep

# For full hardware metrics (slower):
ncu --set full -k regex:slow_kernel -c 1 -o slow_full.ncu-rep python profile_a_kernel.py
```

Open `*.ncu-rep` files in the Nsight Compute GUI for the visual experience:
- Speed of Light page
- Memory Workload Analysis
- Warp State Statistics
- Roofline plot

## What you should see

Both kernels do the same math: sum a 4096×4096 matrix along one axis. Memory access pattern is the only difference.

```
fast_kernel — coalesced row-wise:
  duration:    ~50 µs
  compute SOL: low (just a sum)
  memory SOL:  60-80% — close to HBM peak
  occupancy:   high
  → MEMORY-BOUND but well-tuned

slow_kernel — strided column-wise:
  duration:    ~500 µs (10× slower!)
  compute SOL: low
  memory SOL:  10-20% — terrible utilization
  occupancy:   high (warps are scheduled, just stalled)
  → MEMORY-BOUND and badly tuned

  Top stall reason: "Long Scoreboard" — waiting on HBM
  Achieved bandwidth: ~5% of HBM peak
```

Same arithmetic. Same FLOPs. 10× difference because of how the GPU loads data.

This is the lesson: **two kernels at the same Speed of Light % can have wildly different durations because of the access pattern**. Always look at achieved bandwidth, not just SOL.

## Try

- **Open `slow.ncu-rep` in the GUI**, navigate to "Memory Workload Analysis." Confirm the "L1/TEX Cache" hit rate is very low — uncoalesced reads bypass the coalescing buffer and hammer L2/HBM directly.
- **Modify `slow_kernel` to use a stride-friendly layout**: transpose the input matrix first. The "slow" kernel becomes "fast." Re-profile to confirm.
- **Run with `--set full`** on the slow kernel. Look at "Warp State Statistics" — Long Scoreboard should dominate.
- **Compare to `torch.sum(x, axis=1)`** in the trace. PyTorch dispatches a tuned cuBLAS/cuDNN kernel; should match or beat your fast kernel.
- **Set up a tiny CI pattern**: write a script that runs ncu on `fast_kernel`, parses the report, and prints "FAIL" if memory SOL drops below 50%. This is a real-world regression test pattern.

## When you'd use ncu in real work

| Scenario | Tool |
|---|---|
| "My kernel takes 50ms — why?" | ncu (after nsys identifies which kernel) |
| "Did my optimization actually help at the hardware level?" | ncu before/after |
| "We need to verify a kernel didn't regress in this PR" | ncu in CI via `parse_ncu_report.py` pattern |
| "The whole training loop is slow" | nsys first, then ncu on the worst kernel |

ncu is the second tool you reach for, not the first. Always start with a timeline (nsys/torch.profiler) to find the slow kernel, then zoom in.

## Where this goes

Topic 03 (PyTorch profiler) is the everyday workflow — much faster iteration but lower fidelity than nsys/ncu. Topic 04 (roofline) takes the SOL and bandwidth numbers from this topic and puts them on a single picture.
