# 05 — Accelerator Landscape

## Files

- `CONCEPTS.md` — execution models (SIMT / systolic / static dataflow / wafer / Tensix), per-vendor toolchain map, the five questions to ask of any new accelerator.
- `landscape_table.py` — runnable script that prints the landscape table and the five-question framework. Useful as a desk reference.
- `notes_per_vendor.md` — single-page cheat sheet: what each vendor's compiler is called, what IR it speaks, where to find docs.

## Quickstart

```bash
python landscape_table.py
```

No GPU required. The script just dumps text — it's a study aid, not a benchmark.

## What to look for

- The table groups vendors by execution model, not by company. Notice that Groq and Cerebras are both "static dataflow" but at very different scales (chip vs wafer).
- Every non-NVIDIA stack has a kernel-author DSL: Pallas (TPU), NKI (Trainium), TT-Metal (Tenstorrent). Triton's "win" is being the *common* one for SIMT.
- StableHLO / MLIR shows up in three places: XLA, IREE, Modular. That's the convergence story.

## Try

- Pick one row from the table. Find its compiler's GitHub or docs page (links in `notes_per_vendor.md`). Read the architecture overview only — 30 minutes.
- Write three sentences answering the five questions for that vendor.
- Compare TPU's Pallas with NVIDIA's Triton — same idea, different IR, different memory model. Where do they diverge?

## Where this goes next

- Topic 06 (IREE) is the open-source "one IR, many targets" answer to the fragmentation this topic surveys.
- Topic 07 (CUTLASS / CuTe DSL) goes deep on the NVIDIA row.
- Topic 08 (AI-assisted kernels) is what changes if LLMs can write the per-vendor kernels for us.
