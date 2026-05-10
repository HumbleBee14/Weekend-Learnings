# 03 — PyTorch Profiler

## Files

- `CONCEPTS.md` — torch.profiler vs nsys, the schedule API, Perfetto, memory snapshots, distributed traces, gotchas
- `profile_with_torch.py` — small training loop with the schedule API; emits a Perfetto-ready JSON
- `memory_snapshot.py` — the modern memory profiling path: record_memory_history → dump → visualize

## Quickstart

```bash
pip install torch

# Time profiling
python profile_with_torch.py
# → traces/*.json   open at https://ui.perfetto.dev

# Memory profiling
python memory_snapshot.py
# → snap.pickle     open at https://pytorch.org/memory_viz
```

## What you should see

`profile_with_torch.py` prints a table like:

```
Top GPU consumers
                                Self CUDA   Self CUDA %    CUDA total
ProfilerStep#7                  0.000us           0.00%       7.123ms
  step_7                        0.000us           0.00%       7.123ms
    forward                     0.000us           0.00%       2.450ms
      aten::linear              0.000us           0.00%       2.180ms
        aten::matmul            0.000us           0.00%       2.150ms
          ampere_sgemm_*        2.150ms          30.20%       2.150ms
          ...
    backward                    0.000us           0.00%       3.200ms
    optim_step                  0.000us           0.00%       1.270ms
```

What to look for:
- **Top kernel** by CUDA time. For a small MLP this is the matmul (cuBLAS GEMM).
- **Forward vs backward ratio.** Backward should be ~2× forward (extra grad compute).
- **Optim step cost.** AdamW unfused can be 15-20% of step time; fused (`AdamW(fused=True)`) drops it to single digits.

The Perfetto trace shows the same data on a timeline — easier to spot gaps between kernels (CPU bottlenecks) than from the table.

## Try

- **Open the trace in Perfetto.** Find the `forward` block. Zoom in until you can see individual kernels.
- **Toggle `with_stack=True` off.** Trace becomes much smaller and the profiler runs faster, but you lose the source-line attribution.
- **Add `torch.compile(model)`.** Re-profile. Kernel count should drop (Inductor fused pointwise ops). Look for `triton_*` kernel names.
- **Switch to `AdamW(fused=True)`.** Re-profile. The `optim_step` block should shrink dramatically.
- **Run `memory_snapshot.py`.** Open the pickle in memory_viz. Note the difference between peak allocated (what your tensors use) and peak reserved (what the CUDA allocator holds onto for reuse).

## When you'd use this in real work

Daily iteration: torch.profiler. Quick sanity check on a notebook: torch.profiler. Need to know if your `torch.compile` actually fused things: torch.profiler with the kernel name filter.

When `torch.profiler` runs out: deeper kernel metrics (occupancy, stalls), multi-process traces, or system-level activity → switch to `nsys` (Topic 01) and `ncu` (Topic 02).

## Distributed profiling preview

For multi-GPU training, each rank produces its own JSON. To analyze them together you use **HTA (Holistic Trace Analysis)**:

```python
from hta.trace_analysis import TraceAnalysis
analyzer = TraceAnalysis(trace_dir="./traces/")
analyzer.get_idle_time_breakdown()      # which rank is idle most?
analyzer.get_comm_comp_overlap()        # how well does compute overlap with communication?
analyzer.get_frequent_cuda_kernel_patterns()
```

Topic 06 (Profiling Training) goes deep on this.

## Where this goes

Topics 01-03 are the three profiling tools. Topic 04 is the *framework* you use to interpret what they show — the roofline model. Topics 05 and 06 apply all of this to real LLM serving and training workloads.
