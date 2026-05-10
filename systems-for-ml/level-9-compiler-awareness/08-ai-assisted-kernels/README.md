# 08 — AI-Assisted Kernels

## Files

- `CONCEPTS.md` — what "LLM-written kernel" actually means (three distinct senses), KernelBench results, where LLM kernel generation works in 2026 and where it doesn't, the realistic trajectory.
- `try_llm_kernel.md` — a small, honest exercise: prompt a model to write a Triton softmax, benchmark it against PyTorch, read the failure modes. No GPU? Read the recorded transcript instead.
- `kernelbench_walk.py` — script that fetches KernelBench's problem list locally and prints the difficulty distribution. A study aid for understanding the eval.

## Quickstart

```bash
# Optional: clone KernelBench for the walk script.
git clone --depth=1 https://github.com/ScalingIntelligence/KernelBench.git ~/kernelbench
python kernelbench_walk.py --root ~/kernelbench
```

The exercise in `try_llm_kernel.md` needs an LLM API key and (ideally) a GPU. The recorded-transcript section in the same file is the no-GPU fallback.

## What to look for

- KernelBench's problems are structured: levels 1 (basic), 2 (fused), 3 (full architectures). Notice the long tail at level 3 — those are where current models fail most.
- In your own LLM-kernel attempt, watch for the typical failure modes: silent shape mismatches, off-by-one in reductions, not handling masking, missing the `tl.constexpr` annotations Triton needs.
- The benchmarking harness — compile, run, compare numerics, time — is where most of the engineering effort lives. The LLM call itself is one line.

## Try

- Run `try_llm_kernel.md`'s prompt against two different frontier models. Compare outputs side by side. Note what each gets right and wrong.
- Take a Triton kernel an LLM wrote. Profile it with `ncu` (Nsight Compute) on a real GPU. Where is the perf left on the table — register spills, memory bandwidth, lack of warp specialization?
- Write a one-paragraph note: "If I were building an LLM-driven kernel system today, the harness I'd need is …". This is the productive question. The model is one component.

## Where this goes next

- This is the closing topic of the level. The next step after this awareness pass is the level's writeup (`reports/compiler-tour.md`) and the decision: do you want to specialize in compiler / kernel engineering as a separate track?
- If yes: LLVM Kaleidoscope → MLIR Toy → Triton internals → CUTLASS deep read. That's a months-long path.
- If no: you have enough to read traces, hold the conversation, and pick the right tool when it matters in your normal work.
