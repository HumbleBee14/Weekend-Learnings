# 04 — What MLIR and LLVM Are

## Files

- `CONCEPTS.md` — LLVM vs MLIR cleanly separated; the dialects you'll actually see in ML; progressive lowering as a mental model; common confusions cleared up.
- `walk_dialects.py` — prints canonical snippets in `stablehlo`, `linalg`, `scf+vector`, `gpu/nvgpu`, `llvm` so you can read the shape of each dialect side by side.

## Quickstart

```bash
python walk_dialects.py
```

Read top to bottom. That's the lowering order. Each box in the lowering picture from Topic 01 corresponds to one of these dialects.

## Try

- Install the MLIR/LLVM toolchain (`brew install llvm` on macOS, or build from source). Save one of the snippets to `block.mlir` and run:
  ```bash
  mlir-opt block.mlir --convert-linalg-to-loops --print-ir-after-all
  ```
  to see the actual pass-by-pass transformation.
- Install IREE (`pip install iree-compiler iree-runtime`) and compile a StableHLO module:
  ```bash
  iree-compile block.mlir --iree-hal-target-backends=llvm-cpu -o block.vmfb --mlir-print-ir-after-all 2> dump.log
  ```
  The dump shows every IR snapshot through the lowering chain.
- Open the MLIR Toy tutorial (https://mlir.llvm.org/docs/Tutorials/Toy/). You don't have to implement it. Read Chapters 1, 3, and 5 for the IR design choices.

## Where this goes next

- Topic 05 covers how different accelerators ride this same MLIR substrate to reach their hardware.
- Topic 06 covers IREE — a fully-MLIR-native ML compiler that's the easiest practical entry point.
- Deep-dive: `compiler-and-kernels/level-6-mlir-in-practice/`.
