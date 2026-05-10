# Level 3 — Learning Path

The most important "before vs after" level in the curriculum. Every Level 4 fix is justified by a Level 3 measurement. If you skip this, Level 4 becomes "I tried torch.compile" cargo-culting; with this, it becomes "I added a paged KV cache because my profiler showed 38% of decode time was in cache-lookup with the contiguous layout."

This level produces two artifacts that survive into Project 1's deliverables:
- `reports/profiling-mini-serve.md` (Topic 05)
- `reports/case-study.md` (Topic 07)

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-nsight-systems-basics/` | 2-3h | Read a CUDA timeline, find gaps and serialization, annotate with NVTX |
| `02-nsight-compute-basics/` | 2-3h | Speed-of-Light analysis on a single kernel, Warp State Statistics, the CI integration pattern |
| `03-torch-profiler/` | 1-2h | The everyday profiling tool. Schedule API, Perfetto, memory snapshots |
| `04-roofline-model/` | 1-2h | The framework that makes profiler numbers actionable. Estimate AI on a napkin for any LLM kernel |
| `05-profiling-inference/` | 3-4h | Apply Topics 01-04 to your `mini-serve`. Produces `profiling-mini-serve.md` |
| `06-profiling-training/` | 2-3h | Same toolset for training loops; HTA for multi-rank; FlightRecorder for hangs |
| `07-optimization-case-study/` | 4-6h | End-to-end: take a slow model, fix it 5 times, measure each delta. The capstone. |

## What hardware you need

- **A real NVIDIA GPU.** Free Colab has a T4 — works for everything except some of the more advanced Hopper-specific metrics in `ncu`.
- **A100 or H100** ideally for the inference profiling (Topic 05) — vLLM's full feature set works best there.
- **Nsight Systems and Nsight Compute installed.** Both are free; ship with CUDA toolkit. Bundled NVIDIA driver versions work on Linux. macOS GUIs can open `.nsys-rep` and `.ncu-rep` files generated remotely.

## Each topic folder

Same shape as Levels 1 and 2:

- `CONCEPTS.md` — the theory and 2026 state of the art
- One or more code files (`.py`) demonstrating the tool/technique
- `README.md` — quickstart, expected output, things to try

## The mental model you're building

Levels 1 and 2 taught you to write servers and kernels. Level 3 teaches you to measure them.

The progression in this level:

```
Topic 01: timeline (where is the time going overall)
Topic 02: per-kernel hardware metrics (why is THIS kernel slow)
Topic 03: everyday Python profiler (fast iteration)
Topic 04: roofline framework (compute-bound? memory-bound? overhead-bound?)
Topic 05: applied to inference (your mini-serve)
Topic 06: applied to training
Topic 07: full case study workflow (the muscle memory)
```

By the end, you can take any slow PyTorch workload and produce a credible optimization plan from a profile. Without this skill, every later optimization is a guess.

## After this level

You go into Level 4 (LLM optimization) with two reports in hand and a clear list of bottlenecks to fix. Every Level 4 topic — paged KV, continuous batching, FP8 quantization, kernel fusion, speculative decoding — is justified by a measurement you took here.

The skill the curriculum is building from this point on: **diagnosis before treatment**. Anyone can apply optimizations. Few can look at a slow LLM workload and say "the second matmul in the FFN is at 23% bandwidth utilization; the next move is tiling, not lowering precision."
