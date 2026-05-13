# 05 — Dynamic shapes and recompilation

The single most common reason `torch.compile` "doesn't work" for LLM inference: the seqlen varies every batch, Dynamo recompiles every batch, exceeds `cache_size_limit = 8`, and silently falls back to eager forever. Latency is now strictly worse than not compiling at all.

You will reproduce this failure mode and fix it three different ways.

## Hardware

CUDA GPU needed for meaningful timings. T4 fine. The recompile log itself is GPU-agnostic, but the win/loss numbers only mean something if you have real launch overhead.

## What to run

```bash
# 1. Watch the failure: recompile every step until Dynamo gives up
TORCH_LOGS="recompiles" python recompile_demo.py --mode naive

# 2. Fix with mark_dynamic on the specific seqlen dimension
TORCH_LOGS="recompiles" python recompile_demo.py --mode mark_dynamic

# 3. Fix with dynamic=True (the blunt instrument)
TORCH_LOGS="recompiles" python recompile_demo.py --mode dynamic

# 4. Fix with bucketing (the vLLM approach)
TORCH_LOGS="recompiles" python recompile_demo.py --mode bucket
```

Each mode prints: number of recompiles, mean latency over 50 calls at varying seqlens, and whether Dynamo fell back to eager.

## What you should observe

| Mode | Recompiles | Steady-state latency | Notes |
|---|---|---|---|
| `naive` | 8 then fallback | worse than eager | the cache-limit failure |
| `mark_dynamic` | 1 | best | symbolic shape on seqlen |
| `dynamic=True` | usually 1, sometimes 2-3 | similar to mark_dynamic, sometimes worse | broad-strokes; SymInt resolution can pessimize |
| `bucket` | one per bucket (≤5) | best within bucket, plus padding overhead | what vLLM does for CUDA graph capture |

Then open the FX graph dumps (set `TORCH_LOGS="aot_graphs"` and re-run `mark_dynamic`). Look at the placeholder shapes. You will see `s0` or `arg0_1` for the seqlen dim — that's the symbolic int. Look for shape expressions like `s0 * 1024` in subsequent ops. That's `SymInt` arithmetic.

Write in [`notes.md`](notes.md): the recompile counts for each mode on your hardware, your timing table, and one concrete observation about a SymInt expression in the FX graph.

## Why each fix works

**`mark_dynamic`** tells Dynamo at trace time that this dimension is symbolic. The FX graph carries a `SymInt` for that dim from the start; downstream ops express their shapes in terms of it; no specialization happens; one compile serves every seqlen.

**`dynamic=True`** is the same idea applied to *every* dim. Sometimes that's what you want; sometimes Dynamo can't keep one of the dims symbolic without confusing a downstream kernel grid, and it has to either fall back to a runtime evaluation (cheap) or recompile (expensive). Test both, pick the winner.

**Bucketing** sidesteps the problem: pre-compile for a fixed set of shapes, pad inputs to the nearest bucket. The cost is padding overhead (you do work on padded positions). The benefit is each bucket can be a fully-captured CUDA graph, which Dynamic shapes alone usually cannot do. vLLM's `cudagraph_capture_sizes` is exactly this list.

## Reference

- [Dynamic Shapes guide](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html).
- [Dealing with Recompilations](https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html).
- [Ian Barber: Dynamic Shapes in PyTorch (Apr 2025)](https://ianbarber.blog/2025/04/04/dynamic-shapes-in-pytorch/).
- [vLLM compilation config — `cudagraph_capture_sizes`](https://docs.vllm.ai/en/latest/design/torch_compile/).

## Common pitfalls

- **You forgot to clear the cache between modes.** Dynamo's cache persists in the function object. The script re-creates the compiled function each mode to avoid this; if you adapt the script make sure you do the same.
- **You enabled `dynamic=True` and recompile count went up.** Yes, this happens. `dynamic=True` is not always the win; benchmark both.
- **You captured a CUDA graph with the bucketed approach but it replayed wrong outputs.** Verify each bucket's output against eager once before believing the timings.
