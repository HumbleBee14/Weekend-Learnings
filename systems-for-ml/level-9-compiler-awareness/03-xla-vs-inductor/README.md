# 03 — XLA vs Inductor

## Files

- `CONCEPTS.md` — the two stacks side by side; OpenXLA's 2024–2026 split from Google; concrete differences (fusion, shape, runtime); when you'd see each.
- `dump_stablehlo.py` — emits StableHLO from a tiny JAX function. Falls back to a printed reference module if JAX isn't installed.

## Quickstart

```bash
pip install "jax[cpu]"
python dump_stablehlo.py
```

You should see a `module @block { func.func public @main ... }` with `stablehlo.dot_general`, `stablehlo.logistic`, `stablehlo.multiply` operations.

## Compare to Inductor

Run the script from Topic 02 with `TORCH_LOGS=output_code`. Compare:

- The Inductor output is generated *Triton or C++ source code* — final-form, ready to compile.
- The StableHLO output is *IR* — still going to be lowered further by XLA.

This is the design difference in one observation: Inductor commits to a backend at codegen time; StableHLO is the portable IR that gets specialized later by the XLA compiler (or by IREE, or by a vendor PJRT plugin).

## Try

- Change `(4, 64)` to a dynamic shape via `jax.export` and inspect the polymorphic StableHLO.
- Run the same JAX function on the GPU backend (`pip install "jax[cuda12]"` if you have one). The StableHLO is identical; the compiled HLO and the resulting kernels differ.
- Read the StableHLO spec: https://openxla.org/stablehlo/spec — every op the JAX lowerer can emit is documented there with semantics.

## Where this goes next

- Topic 04 explains MLIR, which is the substrate StableHLO lives in.
- Topic 06 covers IREE — the StableHLO-consuming compiler that targets Vulkan, Metal, CUDA, CPU.
- The deeper StableHLO work lives in `compiler-and-kernels/level-7-stablehlo-and-xla/`.
