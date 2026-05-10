# 02 — torch.compile Internals

## Files

- `CONCEPTS.md` — Dynamo, AOTAutograd, Inductor as separate concerns; what graph breaks cost; the 2026 piecewise CUDA graph pattern.
- `explore_compile.py` — three experiments: clean compile, deliberate graph break, data-dependent control flow.

## Quickstart

```bash
pip install torch
TORCH_LOGS="graph_breaks,recompiles,output_code" python explore_compile.py 2>&1 | tee trace.log
```

CPU is fine. The C++ codegen path is just as informative as the GPU/Triton one for understanding how Inductor schedules kernels.

## What to look for

- `exp1` — should compile once with no breaks. Count the kernels in OUTPUT_CODE; for an attention block you'll see roughly two compute kernels (pre-attention fused, post-attention fused) plus an external SDPA call.
- `exp2` — should print one `Graph break in user code at ... print(...)` line and produce two compiled regions.
- `exp3` — should print a graph break for data-dependent branching, or trigger a recompile on the second call when the branch flips.

## Try

- Add `dynamic=True` to `torch.compile` calls. Re-run and watch Dynamo emit a single graph that handles multiple shapes.
- Set `TORCH_COMPILE_DEBUG=1`. Open `torch_compile_debug/run_*/fx_graph_readable.py` for each compile and read the FX graph as code.
- Replace the SDPA call with hand-written `q @ k.transpose(-2, -1)` + `softmax` + `@ v`. Compare OUTPUT_CODE to the SDPA version — Inductor should fuse most of it.
- Install `depyf` and decompile the Dynamo-emitted bytecode back to Python. Useful when graph breaks are mysterious.

## Where this goes next

- For the kernel-level deep dive, jump to `compiler-and-kernels/level-2-torch-compile-internals/`.
- Topic 03 contrasts this stack with JAX/XLA on the same kind of model.
- Topic 04 explains the MLIR substrate Triton runs on, which is what Inductor's GPU path emits into.
