# 03 — PyTorch Profiler

## What it is

`torch.profiler` is PyTorch's built-in, Python-native profiler. Lower fidelity than Nsight tools but with three big advantages:

- **No separate tool install.** It's in `torch`. Works in any Python env.
- **Round-trip is 10 seconds**, not 5 minutes. Fast iteration in notebooks.
- **It knows your Python.** Maps GPU kernels back to the PyTorch source line that launched them.

The profiler you'll actually use day to day. `nsys` and `ncu` come out when `torch.profiler` isn't enough.

## What's underneath: Kineto

`torch.profiler` is a thin Python wrapper over **Kineto**, PyTorch's low-level profiling library. Kineto uses NVIDIA's CUPTI under the hood — same data source as `nsys`. The fidelity gap isn't because Kineto is missing data; it's because Kineto runs in-process and doesn't capture system-level activity (CUDA API trace, OS scheduler, multi-process collectives) the way `nsys` does.

For most ML perf work, in-process is enough.

## Basic usage

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        with record_function("forward"):
            output = model(input)
        with record_function("backward"):
            output.sum().backward()

# Inspect
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export for visualization
prof.export_chrome_trace("trace.json")
```

`record_function` is the equivalent of `nvtx.range` — it labels semantic boundaries that show up in the visualization.

## The schedule API

For real workloads (training loops, serving), you don't want to profile *everything* — that's GBs of data. Use `schedule` to profile windows of activity:

```python
schedule = torch.profiler.schedule(
    skip_first=10,    # skip warmup steps (especially with torch.compile)
    wait=1,           # idle for 1 step
    warmup=1,         # profile but discard for 1 step
    active=3,         # actually record for 3 steps
    repeat=1,         # do this cycle once
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./traces"),
) as prof:
    for step in range(100):
        train_step(...)
        prof.step()  # tells the profiler "step boundary; advance the schedule"
```

The trace ends up at `./traces/<hostname>_<pid>.<timestamp>.pt.trace.json`.

`skip_first=10` matters most with `torch.compile` — early steps include compilation overhead and aren't representative of steady-state.

## Visualizing — Perfetto, not Chrome

In 2026, `chrome://tracing` is largely obsolete for ML traces (it chokes above ~1 GB).

Use **Perfetto**: https://ui.perfetto.dev. Drag the `.json` file in. Handles 10+ GB traces. SQL-style queries over the trace. Better timeline rendering.

Same JSON format — `export_chrome_trace()` is the canonical name but the file works in Perfetto.

## What you'll see in the trace

Three rows that matter:

```
CPU thread row:    [Python frames] [forward──][loss──][backward──][optim──]
                       ↑
                       Your Python source-level activity

CPU CUDA API row:  [cudaLaunchKernel] [cudaLaunchKernel] [cudaMemcpyAsync] ...
                       ↑
                       What the CUDA driver was doing (often the bottleneck for small ops)

GPU row:           [matmul] [softmax] [add] [layernorm] [matmul] [...]
                       ↑
                       Actual kernel activity on the device
```

If you see big gaps in the GPU row while the CPU is busy: GPU starved (CPU-bound).
If you see kernels running but Python is sleeping: GPU-bound.
If you see attention/MLP kernels with their PyTorch op names: torch.profiler labeled them for you (this is the "with_stack=True" feature — without it, kernels appear with their internal names only).

## Memory profiling — the new way

`profile_memory=True` is still there, but for serious memory analysis the modern path is the **memory snapshot tools**:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run your workload ...
torch.cuda.memory._dump_snapshot("snap.pickle")
```

Drag `snap.pickle` into https://pytorch.org/memory_viz. Interactive timeline showing:

- Every allocation with stack trace
- Peak memory and what was holding it
- Fragmentation visualization
- Reserved vs allocated divergence

This catches OOMs that aren't really "out of memory" — they're "out of *contiguous* memory" because of fragmentation. Common in long-running serving.

## torch.compile profiling

If you're profiling a `torch.compile`d model, the profiler now annotates Triton kernels with their dynamo region. Kernel names look like `triton_poi_fused_add_mul_silu_42` — the suffix tells you what got fused.

What's hidden:
- Graph break boundaries appear as Python re-entry, not as labeled events
- Recompilations show up as cold calls — use `TORCH_LOGS=recompiles` alongside

For deep `torch.compile` debugging, see `compiler-and-kernels` Level 2 (depyf, FX graph inspection).

## Distributed profiling

`torch.profiler` automatically records NCCL kernel ranges (allreduce, allgather, reduce_scatter) with collective name, dtype, and bucket size. Each rank produces its own JSON.

For multi-rank analysis, use **HTA** (Holistic Trace Analysis, also branded as TraceInsight in 2026):

```python
from hta.trace_analysis import TraceAnalysis
analyzer = TraceAnalysis(trace_dir="./traces/")
analyzer.get_gpu_kernel_breakdown()
analyzer.get_idle_time_breakdown()
analyzer.get_comm_comp_overlap()
```

HTA aligns NCCL collectives across ranks → you can spot stragglers. Topic 06 covers this in depth.

## Comparison vs nsys

| Question | torch.profiler | nsys |
|---|---|---|
| Available without install? | Yes (`pip install torch`) | No (separate package) |
| Captures CUDA driver API? | Limited | Full |
| Captures OS-level activity? | No | Yes |
| Multi-process trace? | Per-process JSONs | Single merged trace |
| CUDA Graph node-level detail? | Yes (recent versions) | Yes (with --cuda-graph-trace=node) |
| Python source attribution? | Yes (with_stack=True) | Yes (--python-sampling) |
| Trace size for 30s run | ~100-500 MB | ~1-2 GB |
| Best for | Daily iteration | Deep dives, multi-process |
| GUI | Perfetto (web) | Nsight Systems (desktop) |

## Gotchas

1. **`with_stack=True` adds 10-20% overhead.** Toggle it off when you only need timing.
2. **`skip_first` matters with torch.compile** — first iters include compilation, atypical perf.
3. **`profile_memory=True` slows things considerably.** Use the dedicated memory snapshot path instead for serious memory work.
4. **Forgetting `prof.step()`** — the schedule never advances, profiler never produces output. Common bug.
5. **`export_chrome_trace` is technically not deprecated**, but Perfetto is the preferred viewer. Same file format.

## References

- PyTorch profiler docs — https://docs.pytorch.org/docs/stable/profiler.html
- HTA / TraceInsight — https://github.com/facebookresearch/HolisticTraceAnalysis
- Memory snapshot blog — https://pytorch.org/blog/understanding-gpu-memory-1/
- Profiling torch.compile — https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
- Perfetto UI — https://ui.perfetto.dev
