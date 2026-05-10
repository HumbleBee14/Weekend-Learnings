# 01 — Nsight Systems Basics

## What it is

Nsight Systems (`nsys`) is NVIDIA's system-wide *timeline* profiler. Captures CUDA API calls, kernel launches, memory transfers, NCCL collectives, OS scheduling, Python frames, and CPU activity onto one synchronized timeline. Output: a `.nsys-rep` file you open in the GUI.

In 2026 it's at version 2026.2.x. Bundled with CUDA 13.x. Free.

Mental model: think of `nsys` as `top` + `strace` + a Wireshark for your GPU, all on one zoomable timeline.

## Why timeline view matters

`time.time()` calls tell you a kernel took 5ms. They don't tell you:

- *Why* — was it waiting on a memory transfer? Was it stuck behind another kernel?
- *Where the gaps are* — kernels that finish fast but with big idle stretches between them
- *Whether the GPU was actually busy* — versus the CPU being slow at queuing work
- *Which stream* it ran on, and whether parallel streams overlapped

A timeline view answers all of these in one screen. Most "why is this slow?" debugging is a timeline-shaped problem.

## The four patterns you're looking for in a timeline

```
Pattern 1: KERNEL LAUNCH OVERHEAD (CPU-bound)

CPU stream:    [k1][k2][k3][k4][k5]
GPU stream:    [k1] [k2] [k3] [k4] [k5]
                    ↑    ↑    ↑    ↑   gaps between kernels — GPU is idle
                                        waiting for next launch

Diagnosis: CPU can't queue kernels fast enough; the GPU is starved.
Fix: kernel fusion (torch.compile, custom Triton), CUDA graphs to batch launches.

────────────────────────────────────────────

Pattern 2: SYNCHRONOUS MEMORY COPY

GPU stream:    [k1] [memcpy H2D] [k2]
                       ↑
                       yellow bar — entire GPU stalled while data moves

Diagnosis: a sync H2D or D2H is on the critical path.
Fix: pinned memory + cudaMemcpyAsync; eliminate the transfer entirely if possible.

────────────────────────────────────────────

Pattern 3: STREAM SERIALIZATION

stream 0: [k1][k2][k3][k4][k5][k6]   ← everything on one stream
stream 1: (empty)

Diagnosis: independent work that could run in parallel is serialized.
Fix: multiple CUDA streams (Level 2 Topic 7).

────────────────────────────────────────────

Pattern 4: NCCL STRAGGLER (training only)

rank 0: [compute][allreduce────────]
rank 1: [compute     ][allreduce───]
rank 2: [compute][allreduce────────]
rank 3: [compute───────][allreduce─]
                        ↑
                        rank 3 is slow → everyone waits for it

Diagnosis: one rank straggling makes all ranks wait.
Fix: find the stragglers's cause (slow disk? thermal throttle? bad NIC?).
```

If you can spot these four in 30 seconds of looking at a trace, you're 80% of the way to using `nsys` effectively.

## The 2026 command line

```bash
nsys profile \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --python-sampling=true \
  -o trace.nsys-rep \
  python my_script.py
```

What each flag does:

- **`-t cuda,nvtx,osrt,cudnn,cublas`** — what to trace. `cuda` = CUDA API + kernels. `nvtx` = your annotations. `osrt` = OS calls (semaphores, file I/O). `cudnn`/`cublas` = library-level ranges.
- **`--cuda-graph-trace=node`** — *critical* in 2026. Without this, CUDA graphs (used by vLLM, SGLang, torch.compile) show as one opaque blob. With it, each node inside the graph is visible.
- **`--capture-range=cudaProfilerApi`** — only record between explicit `torch.cuda.profiler.start()` and `stop()` calls in your code. Lets you skip warmup and bound the trace to ~10s.
- **`--python-sampling=true`** — capture CPython call stacks aligned to the GPU timeline. You can see "Python is in `tokenize_batch` while the GPU is idle."
- **`-o trace.nsys-rep`** — output file.

For LLM serving specifically, also add:
```bash
--trace-fork-before-exec=true     # follow into worker subprocesses
VLLM_WORKER_MULTIPROC_METHOD=spawn  # env var so workers are profilable
```

## NVTX — your annotations

`nsys` records what the GPU does. *You* tell it what those activities mean by annotating your Python code:

```python
import torch.cuda.nvtx as nvtx

# Block style
nvtx.range_push("forward")
out = model(input)
nvtx.range_pop()

# Or context manager (cleaner)
with nvtx.range("forward"):
    out = model(input)

# torch.profiler equivalent — emits both Kineto and NVTX events
with torch.profiler.record_function("forward"):
    out = model(input)
```

In the trace, your "forward" range becomes a labeled bar at the top of the timeline. You can stack ranges (forward → encoder → layer 0 → attention → ...). Without NVTX, you see thousands of unlabeled kernels and have to guess which belongs to what.

**Rule of thumb**: annotate at the *layer* level, not the *operation* level. Annotating every `add` and `mul` clutters the trace. Annotating "embedding," "encoder_layer_0," "attention_layer_0," "mlp_layer_0" is just right.

## Reading the trace

Open the `.nsys-rep` in the Nsight Systems GUI. Layout:

```
┌──────────────────────────────────────────────────────────────────┐
│ TIMELINE (zoomable, scrollable)                                  │
├──────────────────────────────────────────────────────────────────┤
│ Threads (CPU)                                                    │
│   Main thread:    [Python] [Python] [Python]      [Python] ...  │
│   Worker thread:  [tokenize] [tokenize] ...                     │
├──────────────────────────────────────────────────────────────────┤
│ CUDA API (host calls into the driver)                            │
│   [cudaLaunchKernel][cudaMemcpyAsync][cudaLaunchKernel]...      │
├──────────────────────────────────────────────────────────────────┤
│ NVTX (your annotations)                                          │
│   [forward──────────────────][backward──────][optim]            │
├──────────────────────────────────────────────────────────────────┤
│ CUDA HW (what actually happened on the GPU)                      │
│   stream 7: [k1][k2][k3]   [k4][k5]    (kernels)                │
│   stream 8:        [memcpy]                                      │
└──────────────────────────────────────────────────────────────────┘
        ↑                                ↑
        time goes →                      a "gap" here = GPU idle
```

The two views you'll spend most time in: **NVTX** (what semantically is happening) and **CUDA HW** (what the GPU actually did). Visual gaps in CUDA HW = GPU was idle. Find those first.

## Gotchas

1. **Trace files explode in size** for long runs. A 30-second LLM serving trace is often 1-2 GB. Use `--capture-range` to bound it.
2. **`osrt` adds real overhead.** If you're profiling for performance numbers, drop `osrt,cudnn` and use minimal `-t cuda,nvtx`. Re-add when investigating.
3. **Without `--cuda-graph-trace=node`**, modern stacks look like a single blob. You'll mis-diagnose.
4. **CPU sampling skews timing.** `--python-sampling=true` slows the workload by 5-15%. Toggle it off when you only need GPU timing.
5. **Profiling needs the right perms.** On most clusters, `sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` once per boot. Containers need `--cap-add=SYS_ADMIN`.
6. **One trace per rank.** Multi-rank distributed runs produce N traces. Use `nsys recipe` to merge them. HTA (Topic 03) is often easier for multi-rank analysis.

## When `nsys` is the right tool vs not

**Use `nsys` when:**
- You need to see CPU and GPU activity on one timeline
- You suspect kernel-launch-overhead, sync waits, or stream serialization
- You want to see how torch.compile / CUDA graphs structured the work
- You're debugging "GPU util is 60%" — `nsys` shows you why

**Reach for `ncu` (Topic 02) instead when:**
- You already know which kernel is slow and need to understand *why*
- You need per-kernel hardware metrics (occupancy, bandwidth utilization, stall reasons)

**Reach for `torch.profiler` (Topic 03) instead when:**
- You want fast iteration without a separate tool install
- You want Python-frame attribution without `--python-sampling`'s overhead
- You're working from a notebook

The trio is complementary: `nsys` for "where is the time going overall," `ncu` for "why is this kernel slow," `torch.profiler` for everyday lightweight checks.

## References

- Nsight Systems release notes (2026.2.x) — https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html
- "Navigating Nsight Systems" — Henry Ko's walkthrough — https://henryhmko.github.io/posts/profiling/profiling.html
- vLLM profiling guide — https://docs.vllm.ai/en/stable/contributing/profiling/
- NVTX docs — https://nvidia.github.io/NVTX/
