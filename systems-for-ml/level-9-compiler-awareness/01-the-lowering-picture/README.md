# 01 — The Lowering Picture

## Files

- `CONCEPTS.md` — the PyTorch and JAX lowering chains, where MLIR sits, where things go wrong, JIT vs AOT.
- `inspect_lowering.py` — runs a tiny `Linear→SiLU→Linear` block through `torch.compile` with every log channel on. Read the output top-to-bottom: module source, FX graph, AOT post-grad graph, generated kernel.

## Quickstart

```bash
pip install torch
python inspect_lowering.py 2>&1 | less
```

On CPU you see C++ codegen. On GPU you see Triton. The chain is identical above codegen.

## What to look for in the output

- `GRAPH` — Dynamo's FX graph. Plain Python-looking ops on tensors.
- `AOT_GRAPHS` — joint forward/backward graph after decomposition. More ops, smaller ops.
- `OUTPUT_CODE` — the actual generated kernel source. This is what runs.
- `max |eager - compiled|` — sanity check. Should be ~1e-6 or smaller.

Skim the OUTPUT_CODE block in particular. The two `Linear` calls and the `SiLU` should fuse into a single kernel — count the loops.

## Try

- Insert `print(x.shape)` mid-forward. Rerun. You'll see a `graph_break` log entry and two compiled regions instead of one.
- Replace `forward` with a control-flow branch on tensor value (`if x.sum() > 0`). Watch Dynamo bail to a guard.
- Set `TORCH_COMPILE_DEBUG=1` and inspect the dumped artifacts in `torch_compile_debug/`.
- On a GPU box, rerun and read the generated Triton. The `tl.program_id`, `tl.load`, `tl.store` pattern is visible.

## Where this goes next

- Topic 02 zooms into the Dynamo + Inductor pair specifically.
- Topic 03 contrasts this with JAX/XLA on the same kind of model.
- Topic 04 explains the MLIR substrate the Triton compiler runs on.
