# 02 — Nsight Compute Basics

## What it is

Nsight Compute (`ncu`) is NVIDIA's *per-kernel* deep profiler. Where `nsys` (Topic 01) shows you the timeline, `ncu` zooms into one specific kernel and reports: hardware utilization, occupancy, memory throughput, L2 hit rate, warp stall reasons, register pressure.

Microscope-level. You don't run it on a whole workload — you pick the slow kernel from `nsys` and zoom in.

In 2026 it's at version 2026.1.x. Same toolchain bundle as `nsys`.

## The one mental model

`ncu` answers two questions for any kernel:

1. **What is it bottlenecked on?** Compute? Memory? Latency? Sync?
2. **How close to peak hardware capability is it running?**

Everything else in `ncu` is detail supporting those two.

## The Speed-of-Light page

The headline view. Two bars side by side:

```
Compute (SM) Throughput   ████████████████░░░░  78% of peak
Memory Throughput         █████░░░░░░░░░░░░░░░  22% of peak
```

Reading it:

| Compute SOL | Memory SOL | Diagnosis | Optimization direction |
|---|---|---|---|
| > 80% | < 50% | **Compute-bound** | Lower precision (FP16 → FP8), better algorithm, fewer ops |
| < 50% | > 80% | **Memory-bound** | Reduce HBM traffic: tiling, fusion, smaller working set |
| > 60% | > 60% | **Balanced** | You're using the hardware well; gains are smaller |
| < 30% | < 30% | **Latency-bound** | Low occupancy, sync waits, or kernel-launch overhead |

For LLM kernels you'll mostly see memory-bound (decode, RMSNorm, softmax) or balanced (matmul on Hopper). Latency-bound is the surprise category — when the SOL bars are both low and the kernel is *still* slow, the answer is in Warp State Statistics.

## Occupancy — the misunderstood metric

Occupancy = (active warps per SM) / (theoretical maximum warps per SM). Often quoted as a percentage.

Naive interpretation: "higher = better." Wrong in 2026.

**For memory-bound kernels:** high occupancy (≥ 50%) is good. Many warps in flight = the SM can hide HBM latency by switching warps.

**For tensor-core kernels (FlashAttention, GEMM):** *low* occupancy is often optimal. WGMMA + TMA hide latency without needing many warps. FA3 runs at ~2 warps/SM and saturates tensor cores. Chasing "fix your occupancy" advice on a tensor-core kernel will waste your time.

The real diagnostic: look at the SOL bars first. If they're high, occupancy is fine regardless of its number.

## Warp State Statistics — why the kernel is stalled

When a warp is *stalled* (not progressing), `ncu` records why. The dominant stall reason is your bottleneck.

The vocabulary (memorize the top 4):

| Stall name | What it means | What to fix |
|---|---|---|
| **Long Scoreboard** | Waiting on global memory (HBM) | Reduce HBM traffic, fuse reads, better access patterns |
| **Short Scoreboard** | Waiting on shared memory or MIO | Bank conflicts, vectorize, swizzle |
| **Wait** | Fixed-latency dependency (FFMA chains) | Increase ILP, separate dependent ops |
| **Barrier** | At a `__syncthreads()` | Threads imbalanced; some finish work later |
| **MIO Throttle** | SMEM/L1 instruction issue saturated | Reduce SMEM ops or vectorize them |
| **LG Throttle** | Global memory issue saturation | Coalesce, use vec4 loads |
| **Tex Throttle** | Texture instruction throttle | Rare in ML kernels |

Hopper added: **TMA Stall** (waiting on TMA async copy).
Blackwell added: **TMEM Stall** (waiting on tensor memory access).

90% of LLM kernel stalls are **Long Scoreboard** = HBM bandwidth bound. The fix isn't a kernel change — it's a *data access pattern* change.

## The workflow

Three modes, listed by speed:

```
ncu --set basic -k regex:my_kernel -c 1 ./bench    # ~1.5x slowdown of the kernel
ncu --set full  -k regex:my_kernel -c 1 ./bench    # ~50x slowdown of the kernel
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.sum,... ./bench   # surgical, fast
```

- **`--set basic`**: Speed of Light + Launch Stats. Use first. Tells you immediately if it's compute or memory bound.
- **`--set full`**: everything (~150 metrics). Use when `--set basic` says "memory bound" and you want to know exactly which level (L1/L2/HBM).
- **`--metrics ...`**: surgical, used in CI — pick the 3-5 metrics you care about.

The `-k regex:my_kernel` filter is critical. ML workloads launch thousands of kernels per second. `--set full` on every kernel takes hours. Always filter.

The `-c 1` flag captures only one invocation per matching kernel. ML kernels run thousands of times in a workload; you don't need to profile all of them.

## How to find the right kernel name

From your `nsys` trace:

```bash
nsys stats --report cuda_gpu_kern_sum trace.nsys-rep | head -20
```

Lists the kernels by total time. Top kernel name → use as the `-k` regex for `ncu`.

Or in the GUI: right-click a kernel in the timeline → "Show in Statistics" → copy name.

## The Memory Workload Analysis page

For memory-bound kernels (Long Scoreboard stalls), this is the next stop. It shows:

```
HBM read  ████████████   78% of HBM peak bandwidth     1.2 TB/s out of 1.94 TB/s
HBM write ███░░░░░░░░░   22% of HBM peak

L2 hit rate                        85%
L1 (SMEM) hit rate                 92%
L2 read throughput                 4.1 TB/s
SMEM bank conflicts (load)         0
SMEM bank conflicts (store)        12% of stores
```

Diagnosis flow:
- L2 hit rate < 50%? Your access pattern doesn't reuse — fix tiling.
- HBM bandwidth at 80%+ peak? You're memory-bound *and well-tuned*. Only fix is reducing total bytes (fusion, quantization).
- SMEM bank conflicts > 5%? Use swizzled layouts (Triton handles this; CUDA C++ doesn't).
- Big read/write asymmetry? Maybe redundant reads (would benefit from caching) or redundant writes (would benefit from accumulating in registers).

## Roofline view

`ncu --set full` (or the GUI's "Roofline" section) plots your kernel as a single point on the roofline plot:

```
Performance (TFLOPS)
     │
peak ┤────────────────────  compute ceiling
     │
     │           ╱       ← any kernel above this line is theoretically impossible
     │          ╱
     │      ╱──    ← memory bandwidth ceiling slope
     │   ╱
     │ ╱           ← your kernel sits here
     │
     └─────────────────────  Arithmetic intensity (FLOPs/byte)
       low                 high
       memory-bound        compute-bound
```

The point's vertical distance from the roofline ceiling = your headroom. If your kernel is at the ceiling, more tuning won't help — you've maxed out either compute or bandwidth. If it's far below, there's room.

Topic 04 deepens the roofline mental model.

## Modern integration: ncu in CI

Big change in 2025-2026: `ncu` has a Python report API. Read `.ncu-rep` files in CI to detect kernel regressions.

```python
# pip install ncu-report
from ncu_report import load_report

report = load_report("baseline.ncu-rep")
for kernel in report:
    sol = kernel["sm__inst_executed.avg.pct_of_peak_sustained_elapsed"]
    print(f"{kernel.name}: {sol:.1f}% SOL")
```

Pattern in CI:
1. Run `ncu --set basic -k regex:^critical_kernel$ -c 1 -o ci.ncu-rep ./bench`
2. Parse `ci.ncu-rep`, extract Speed of Light %.
3. Compare to a baseline file checked into the repo.
4. Fail the build if SOL drops > 5%.

This is what serious ML systems teams do for kernel performance regression testing.

## Gotchas

1. **Replays.** `ncu` re-runs each kernel many times to gather all metrics (one replay per metric group). Stateful kernels (RNG, KV cache writes) need `--replay-mode application` to be correct, but that's slower.
2. **Permissions.** Same as `nsys`: `sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` once. Containers need `--cap-add=SYS_ADMIN`.
3. **The Hopper/Blackwell metric names changed.** Old guides reference Ampere-era metric names. Check `ncu --query-metrics | grep <thing>` if a metric isn't found.
4. **Don't compare absolute throughput numbers across GPUs.** Same kernel runs at different SOL on different hardware because peaks differ. Compare relative SOL %.
5. **First time the kernel is captured includes JIT overhead.** Always warm up before profiling.

## When `ncu` is the right tool

**Use `ncu` when:**
- `nsys` told you which kernel dominates and you need to know *why*
- You want hardware-level metrics (occupancy, bandwidth, stall reasons)
- You're optimizing a single kernel and need before/after numbers
- You're setting up CI regression testing for kernel performance

**Don't use `ncu` for:**
- Whole-workload profiling — use `nsys`
- Quick iteration in a notebook — use `torch.profiler`
- Multi-GPU collective analysis — use `nsys` + HTA

## References

- Nsight Compute Profiling Guide — https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Nsight Compute 2026.1 features — https://developer.nvidia.com/nsight-compute-2026_1-new-features
- Python Report Interface — https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html
- Kernel Profiling Guide — https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-profiling-guide
