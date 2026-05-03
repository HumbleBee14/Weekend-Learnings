# Level 3 — GPU Profiling & Bottleneck Analysis

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: profiles `mini-serve` from Project 1; the case study from Topic 07 becomes evidence for the Level 4 fixes.

## Week goal

By Friday you can take *any* PyTorch workload — your `mini-serve`, a HuggingFace fine-tune, a training loop — and answer four questions with evidence:

1. Is this compute-bound or memory-bound? *(roofline)*
2. Where exactly is the time going? *(timeline trace)*
3. What's the slowest single kernel and why? *(kernel-level metrics)*
4. What's the next change that would actually help? *(hypothesis from data, not vibes)*

Profiling moves *before* optimization on purpose. Every Level 4 optimization will be justified by a Level 3 measurement, not a guess. If you optimize without profiling first, you're guessing — that's the difference between a junior who "tried torch.compile" and a senior who "saw a 14% MFU gap, traced it to a graph break in the streamer thread, and fixed it for a 1.4× win."

## Where this fits

- **Comes after:** Level 1 (you have a server to profile), Level 2 (you understand the GPU model well enough to interpret a trace).
- **Comes before:** Level 4. The optimization week is downstream of this one — you measure first, then fix.
- **Project this feeds:** Project 1 directly (you'll profile `mini-serve` and document the bottleneck), Project 3 indirectly (training profiling reappears in Level 6).

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | nsight-systems-basics | Capture and read a GPU timeline |
| 02 | nsight-compute-basics | Per-kernel metrics — occupancy, bandwidth, SM util |
| 03 | torch-profiler | PyTorch profiler + Chrome trace viewer |
| 04 | compute-vs-memory-bound | Roofline model |
| 05 | profiling-inference | Profile `mini-serve`, find the slowest component |
| 06 | profiling-training | Profile a training loop |
| 07 | optimization-case-study | Slow model → profile → hypothesis → fix → measure |

### 01 — `nsight-systems-basics`

**What it is.** NVIDIA Nsight Systems (`nsys`) — a system-wide timeline profiler. Captures CUDA API calls, kernel launches, memory transfers, NCCL collectives, and CPU activity on one timeline. Output is a `.nsys-rep` file you open in the Nsight Systems GUI.

**Why it matters.** A timeline tells you what you *cannot* see from `time.time()` calls: gaps between kernels (CPU launch overhead), unexpected D2H transfers (Python touching a GPU tensor), kernel serialization on a single CUDA stream, NCCL waits on stragglers. Most "why is this slow?" answers are visible in 30 seconds of timeline.

**Build steps.**
1. Install Nsight Systems on the GPU host (or use Colab and download the report). `nsys --version` to confirm.
2. Wrap your inference call: `nsys profile -o trace.nsys-rep -t cuda,nvtx --force-overwrite=true python run.py`.
3. Use NVTX markers in your Python: `torch.cuda.nvtx.range_push("forward")` / `range_pop()`. These show up as labeled blocks in the timeline.
4. Open the report in the Nsight Systems GUI. Look for: kernel gaps, long memcpy, single-stream serialization.

**What to look for.** Three classic patterns:
- **Kernel launch overhead** — many tiny kernels with gaps between them. Fix: kernel fusion (`torch.compile`, custom Triton).
- **Synchronous H2D/D2H** — yellow `cudaMemcpy` blocks on the timeline. Fix: pinned memory, async transfers, or eliminating the transfer entirely.
- **Serialization** — kernels run one after another on a single stream when they could run in parallel. Fix: multiple CUDA streams (rarely needed for inference, common in training pipelines).

### 02 — `nsight-compute-basics`

**What it is.** Nsight Compute (`ncu`) — per-kernel deep profiler. For one specific kernel, it reports: achieved occupancy, SM utilization, memory throughput, L2 cache hit rate, warp stall reasons. This is microscope-level. You don't run it on a whole workload — you pick a slow kernel from `nsys` and zoom in.

**Why it matters.** When `nsys` says "this kernel takes 60% of your time," `ncu` tells you why. The two answers that matter most: (a) is it compute-bound or memory-bound? (b) what's the dominant warp stall reason?

**Build steps.**
1. From your `nsys` trace, find the slowest kernel name.
2. `ncu --set full -k "kernel_name_regex" -o detailed python run.py`. The `-k` filter is critical — full profile of every kernel takes hours.
3. Open in the Nsight Compute GUI. Three pages matter: GPU Speed of Light (compute % vs memory %), Occupancy, Warp State Statistics.

**What to look for.**
- **Compute-bound** — Speed of Light shows compute > 80%, memory < 50%. Optimization: precision (FP16/FP8), kernel fusion, better algorithm.
- **Memory-bound** — memory > 80%, compute < 50%. Optimization: tiling, fewer round-trips, FlashAttention-style fusion.
- **Latency-bound** — both metrics low, occupancy low, dominant stall is "long scoreboard" or "barrier." Optimization: more threads in flight, less synchronization.

### 03 — `torch-profiler`

**What it is.** PyTorch's built-in profiler. Lower friction than `nsys` because it lives in your Python code — you wrap a code block with `torch.profiler.profile(...)` and get a Chrome-trace-format JSON. Open it with `chrome://tracing` or `ui.perfetto.dev` and you see the same kind of timeline as Nsight Systems, mapped back to your Python source lines.

**Why it matters.** This is the profiler you'll actually use day-to-day. Lower fidelity than Nsight (no CUDA API detail, no SM occupancy), but the round-trip is 10 seconds instead of 5 minutes, and it shows you Python frames inline with kernels. For "is this bottleneck in my dataloader or my model?" it's the right tool.

**Build steps.**
```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./traces"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        run_one_inference_step()
        prof.step()
```

Open `./traces/*.json` at [ui.perfetto.dev](https://ui.perfetto.dev). Look for the GPU row.

**What to look for.** Same patterns as `nsys`, plus: long Python frames between kernels (your CPU is the bottleneck), unexpected `aten::copy_` calls (silent device transfers), `cudaStreamSynchronize` waits longer than expected.

### 04 — `compute-vs-memory-bound`

**What it is.** The roofline model. Plot a kernel's *arithmetic intensity* (FLOPs per byte read from HBM) on the X-axis and its *achieved performance* (TFLOPS) on the Y-axis. The "roofline" is two lines: a horizontal line at the GPU's peak compute, and a sloped line at the GPU's peak memory bandwidth × intensity. Any kernel sits underneath the roofline. Where it sits tells you the bottleneck.

```
Performance (TFLOPS)
     │   ╱─────────  peak compute (compute-bound regime)
     │  ╱
     │ ╱  ← memory-bound regime (slope = HBM bandwidth)
     │╱
     └────────────── Arithmetic intensity (FLOPs/byte)
```

**Why it matters.** Different bottlenecks need different fixes. Compute-bound: lower precision or better algorithm. Memory-bound: reduce HBM traffic. Latency-bound: more parallelism. The roofline tells you which optimization category is even relevant *before* you spend a day on the wrong one.

**Build steps.**
1. For your slow kernel, compute arithmetic intensity. Matmul (M,K)×(K,N) = 2MNK FLOPs, reads 2(MK + KN) bytes at FP16, writes 2MN. Intensity = `2MNK / (2(MK+KN+MN)·2)` ≈ `K/2` for square matrices in FP16.
2. Measure achieved TFLOPS: `total_flops / kernel_time`.
3. Plot the point on your GPU's roofline (T4: 65 TFLOPS FP16 / 320 GB/s; A100: 312 TFLOPS / 2 TB/s; H100: 990 TFLOPS / 3.4 TB/s for BF16).
4. The point's vertical distance from the roofline is your headroom.

**Insight.** LLM decode (single-token autoregressive generation) is *almost always* memory-bound — you read the entire model weights to generate one token. Prefill (processing the prompt) is compute-bound. This single fact explains 80% of why prefill and decode get separated in modern engines (chunked prefill, disaggregated serving).

### 05 — `profiling-inference`

**What it is.** Apply the tools to `mini-serve`. Generate load with your Locust harness from Level 1, profile with `torch.profiler`, find the bottleneck.

**Build steps.**
1. Run `mini-serve` under torch.profiler with 16 concurrent users for 30 seconds.
2. Open the trace at perfetto. Identify: (a) which kernel takes the most cumulative time; (b) what fraction of wall time is GPU vs CPU; (c) is there padding waste in your batched forward pass?
3. Pick the slowest kernel. Run `ncu` against it specifically. Document Speed-of-Light numbers.
4. Compute arithmetic intensity for that kernel; place it on a roofline; identify the regime.

**Output.** A page in `reports/profiling-mini-serve.md` with: timeline screenshot (annotated), top-3 kernels by time, roofline placement, hypothesis for what to fix in Level 4. This is the *justification* for Level 4's optimizations — without this, Level 4 is just "I tried torch.compile."

### 06 — `profiling-training`

**What it is.** Same tools, applied to a training loop. The bottlenecks are different: data loading, gradient sync (NCCL), optimizer step, sometimes the loss computation.

**Build steps.**
1. Take a small training script — fine-tuning Qwen2.5-0.5B on a tiny dataset is enough.
2. Profile 50 steps with `torch.profiler`.
3. Look for: dataloader gaps (CPU rows show idle GPU during data fetch), all-reduce time as a fraction of step time (if multi-GPU), optimizer step time.
4. Document the bottleneck.

**Insight to carry.** Training is dataloader-bound more often than people admit. If your GPU sits idle 30% of the time waiting for the next batch, no optimization to the model will help — you need pinned memory, more dataloader workers, or a faster tokenizer. This connects directly to Level 6's data-pipeline-throughput graph.

### 07 — `optimization-case-study`

**What it is.** End-to-end: take a deliberately slow PyTorch model, profile it, form a hypothesis about the bottleneck, fix it, measure the improvement. This is the muscle memory you're building all week.

**Build steps.**
1. Start with a small fine-tune script that's clearly slow (e.g., uses `eager` attention instead of SDPA, or has no `torch.compile`).
2. Profile. Identify bottleneck.
3. Apply *one* fix. Re-profile. Measure delta.
4. Repeat with a second bottleneck.
5. Write up: starting throughput, three fixes applied, ending throughput, what each fix moved on the roofline.

**Output.** `reports/case-study.md` — the systems-paper format from the outer README.

## Project work this week

You're not closing a project, but you're producing **the evidence** that justifies Level 4's fixes:

```
mini-serve/
└── reports/
    ├── week1.md                        # from Level 1
    ├── profiling-mini-serve.md         # NEW — timeline, kernels, roofline placement
    └── case-study.md                   # NEW — slow model → fixes → measured deltas
```

Level 4 will reference `profiling-mini-serve.md` directly. "I added a paged KV cache because my profiler showed 38% of decode time was in cache lookup with the contiguous layout" is a sentence you can only write if Level 3 happened first.

## Definition of done

- [ ] You can capture a Nsight Systems trace and identify the dominant time category in <2 minutes.
- [ ] You can run Nsight Compute against a single kernel and read its Speed-of-Light page.
- [ ] You can use `torch.profiler` for fast iteration loops.
- [ ] You can place a kernel on a roofline and state the regime (compute / memory / latency).
- [ ] You have a written profiling report for `mini-serve` with timeline, kernel breakdown, roofline.
- [ ] You have one end-to-end case study: starting numbers → diagnosed bottleneck → fix → ending numbers.

## Resources (canonical only)

- **Nsight Systems user guide** — [docs.nvidia.com/nsight-systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html). Read "Profiling from the CLI."
- **Nsight Compute user guide** — [docs.nvidia.com/nsight-compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html). Read "Kernel Profiling Guide."
- **PyTorch Profiler tutorial** — [pytorch.org/tutorials/recipes/recipes/profiler_recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
- **Roofline model paper** — Williams, Waterman, Patterson 2009. The original. [Direct PDF](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).
- **Horace He — "Making Deep Learning Go Brrrr"** — [horace.io/brrr_intro](https://horace.io/brrr_intro.html). Best single-page intro to compute-vs-memory-bound thinking.
- **PyTorch Trace Analysis (HTA)** — [github.com/facebookresearch/HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis). Useful for distributed traces in Level 6.

## Common pitfalls

1. **Profiling without warmup.** First iteration includes JIT, allocator warmup, kernel cache miss. Always discard the first N iterations.
2. **Profiling with `torch.compile` enabled before understanding the eager baseline.** Compile changes the kernel landscape. Measure eager first, then compiled, then compare.
3. **Trusting `time.time()` around CUDA calls.** CUDA is async. You need `torch.cuda.synchronize()` before the second `time.time()`, or use `cudaEvent`s. Otherwise you're timing the launch, not the work.
4. **Ignoring the CPU side.** If your dataloader is the bottleneck, no GPU optimization will help. Look at the CPU row of every trace.
5. **Optimizing the wrong kernel.** A kernel that takes 5% of time and is 100% slow is a smaller win than a kernel that takes 60% of time and is 30% slow. Always sort by total time, not by single-call time.
6. **Skipping the roofline because it feels academic.** It's not. It's the single best tool for picking which optimization category to even try.

## What you'll be able to do after this week

> Profile an LLM inference server end-to-end with Nsight Systems, Nsight Compute, and PyTorch Profiler. Place a workload on a roofline plot. Identify memory-bound decode and compute-bound prefill regions. Produce an optimization plan grounded in measured kernel-level data — not in vibes.

The skill here is *diagnosis*, not consumption. Anyone can run vLLM. Far fewer can look at a slow LLM workload and say "the second matmul in the FFN is at 23% memory-bandwidth saturation; the next move is tiling, not lowering precision."
