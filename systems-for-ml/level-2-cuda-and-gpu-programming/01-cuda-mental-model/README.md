# 01 — CUDA Mental Model

## Files

- `CONCEPTS.md` — the execution hierarchy (grid → block → warp → thread), SIMT, memory hierarchy preview, compute capability
- `GPU-NOTES.md` — pocket reference: SM, warp, tensor cores, HBM, 2026 hardware quick numbers
- `diagram_warps.py` — small Triton kernel that prints the warp/block topology of one launch

## Quickstart

```bash
pip install triton torch
python diagram_warps.py     # needs a GPU (free Colab T4 works)
```

You'll see something like:

```
Launched grid=4 block=128 → 512 threads total
That's 16 warps.

Block 0:
  Warp 0: threads 0..31
  Warp 1: threads 32..63
  Warp 2: threads 64..95
  Warp 3: threads 96..127

Block 1:
  Warp 0: threads 128..159
  ...
```

That's the topology you draw on a whiteboard from now on.

## What you should be able to do

- Sketch grid → block → warp → thread on paper from memory
- Explain what SIMT means and why warp divergence matters
- Name the four memory levels (registers, SMEM, L2, HBM) and their rough bandwidth
- Identify a GPU's compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`) and know what features that enables

## Skip code, do drawing

The point of this topic isn't to write code. It's to lock down the mental model. Spend 30 minutes drawing the hierarchy on paper, putting in real numbers (132 SMs on H100, 32 threads/warp, 228 KB SMEM per SM). The drawing is the artifact; the script is just to confirm the warps-of-32 fact in your own data.

## Where this goes next

Topic 2 puts the model into actual CUDA C++ code: vector add, ReLU, softmax. The mental model from this topic carries through everything else.
