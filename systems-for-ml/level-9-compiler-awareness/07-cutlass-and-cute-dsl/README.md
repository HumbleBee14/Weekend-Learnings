# 07 — CUTLASS and CuTe DSL

## Files

- `CONCEPTS.md` — what CUTLASS is, why CuTe replaced the old template towers, where this fits between Triton and cuBLAS, who actually uses it (FlashAttention, FlashInfer, TRT-LLM, DeepGEMM).
- `inspect_cutlass.py` — clones (or scans) the CUTLASS repo and prints a guided reading order: the files most worth opening first.
- `cute_layout_walk.md` — annotated walkthrough of CuTe's layout algebra worked through three concrete examples (row-major tile, swizzled SMEM, warp fragment).

## Quickstart

```bash
git clone --depth=1 https://github.com/NVIDIA/cutlass.git ~/cutlass
python inspect_cutlass.py --root ~/cutlass
```

No GPU required. This is reading and inspection — the kernel-writing exercise is months of work, not the goal of this awareness pass.

## What to look for

- The repo is large. The `inspect_cutlass.py` script prioritizes ten files: the main README, three CuTe doc pages, two tutorial examples, and three production-grade kernels (one Hopper SGEMM, one FlashAttention reference, one FP8 GEMM).
- Watch the difference between `cutlass/include/cutlass/gemm/` (older C++ template kernels) and `cutlass/include/cute/` plus `cutlass/python/` (the CuTe DSL). The new code is markedly shorter.
- Open one production kernel from FlashInfer or DeepGEMM after reading the CUTLASS examples. Same idioms, real workload.

## Try

- Run the script. Read the top three files it points to. 1 hour total.
- Look at one CuTe `Layout` example and reproduce its `(coord) -> offset` mapping by hand on paper. The algebra clicks once you do it once.
- Find one `cp.async.bulk` (TMA) call in a CUTLASS kernel. Trace what it's loading and into what shared-memory layout. This is the Hopper-era pattern.
- Compare a Triton matmul (e.g., the canonical Triton tutorial) with a CUTLASS SGEMM example. Same mathematical operation, two very different IRs. Note where each gives up control.

## Where this goes next

- Topic 08 is the meta-question: now that LLMs can attempt to write these kernels, what changes?
- For real depth on this track, the path is: CuTe quickstart → CuTe layout algebra → MLIR Toy → reading FlashAttention's CUTLASS source → contributing a small kernel back. That's months. Not this week.
