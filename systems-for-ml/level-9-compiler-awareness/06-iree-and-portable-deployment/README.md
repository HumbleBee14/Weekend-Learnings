# 06 — IREE and Portable Deployment

## Files

- `CONCEPTS.md` — IREE's dialect chain (StableHLO → Linalg → Flow → Stream → HAL), how its runtime differs from ONNXRuntime, where it fits and doesn't in 2026.
- `import_and_inspect.py` — minimal PyTorch → StableHLO export via `iree-turbine`. Dumps the MLIR for a tiny model; readable in any text editor.
- `dialect_walk.md` — annotated walkthrough showing one matmul as it lowers through each IREE dialect, with line-pointers into the IREE repo.

## Quickstart

```bash
pip install iree-turbine torch
python import_and_inspect.py
# emits: model.mlir  (StableHLO)
#        model.vmfb  (compiled IREE artifact, CPU backend by default)
```

CPU backend works without a GPU. To target Vulkan or CUDA, pass `--device-list=vulkan` or `cuda` to the compile call (see the script).

## What to look for

- `model.mlir` is StableHLO. Find the `stablehlo.dot_general` (the matmul) and the `stablehlo.logistic` (sigmoid inside silu). That's the vendor-neutral form.
- `model.vmfb` is a binary container. Run `iree-dump-module model.vmfb` (ships with IREE tools) to see the per-target compiled kernels inside.
- The artifact is the same regardless of which backend you run it on later — that's the "compile once" claim, modulo the target list you compiled for.

## Try

- Recompile with `--device-list=vulkan` (Linux/Android) or `metal` (macOS). Same script, same model, different backend.
- Open `model.mlir` and edit one op (e.g., change `silu` to `gelu` upstream and re-run). Watch the StableHLO change shape.
- Use the `iree-compile` CLI directly on `model.mlir`. The CLI is the same tool the Python API drives — useful for embedded build pipelines.
- Compare an IREE-compiled artifact's perf against `torch.compile` for the same tiny model on CPU. IREE will often be faster on CPU because Inductor's CPU backend is less invested in than the GPU one.

## Where this goes next

- Topic 07 (CUTLASS / CuTe DSL) is the other end of the spectrum: vendor-specific, hand-tuned, not portable, but speed-of-light on NVIDIA.
- The MLIR Toy tutorial (https://mlir.llvm.org/docs/Tutorials/Toy/) is the right next step if you want to write your own dialect. Comes after this awareness pass, not during it.
