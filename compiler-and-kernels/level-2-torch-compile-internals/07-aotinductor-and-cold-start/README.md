# 07 — AOTInductor and the cold-start story

`torch.compile` is JIT. First call traces, compiles, autotunes, and emits Triton. For a long training run this is amortized. For multi-model serving (vLLM hosting 30 fine-tunes, autoscaling, serverless inference) it is fatal — minutes of cold-start make the whole feature unusable.

**AOTInductor** is the answer. You run `torch.export` once, ship a `.pt2` archive, and at deploy time load it in milliseconds.

## Hardware

T4 is fine. You need CUDA for the cold-start numbers to be representative (most of the cold-start cost is Triton compilation + autotune).

## What to run

```bash
# 1. Compile and export the model AOT. Produces ./packaged_model.pt2
python export_and_compile.py

# 2. Measure cold-start: load the .pt2 and run one inference
python load_and_run.py

# 3. Compare to JIT torch.compile cold-start
python cold_start_compare.py
```

## What you should observe

| Path | Cold-start (first inference) | Steady-state |
|---|---|---|
| Eager | ~50–200 ms | baseline |
| `torch.compile` JIT first call | 10–60 s | 1.3–1.8× faster |
| AOTInductor `.pt2` load + first call | < 1 s | matches JIT |

The exact JIT cold-start scales with model size and shape variety. A 1B param model with one shape: ~20 s. A 7B model with 5 shape buckets: minutes.

Write in [`notes.md`](notes.md): your numbers, and the size of the `.pt2` archive. If the archive is large (>2× the weights size), you have the autotuned PTX bundled — that's fine for one model, painful for multi-model serving where you want to swap weights without re-shipping kernels.

## What's different about `torch.export`

`torch.export` is strict. No graph breaks allowed. The trace must produce one whole FX graph or export fails. This is why sub-modules 03 (fix breaks) and 05 (handle dynamic shapes) are prerequisites:

- Any `print`, any `.item()`, any data-dependent `if` will fail export. Fix them first.
- Dynamic shapes need explicit `Dim` declarations:
  ```python
  from torch.export import Dim
  batch = Dim("batch", min=1, max=64)
  seqlen = Dim("seqlen", min=1, max=2048)
  ep = torch.export.export(
      model, (example,),
      dynamic_shapes={"x": {0: batch, 1: seqlen}},
  )
  ```
  If you don't declare a dim dynamic, it's specialized to the example shape.

After `export`, you call `torch._inductor.aoti_compile_and_package(ep, args, package_path=...)`. The archive contains the compiled `.so` (or Triton PTX cache), the FX graph, the dynamic-shape constraints, and (optionally) the weights.

At load time:

```python
loaded = torch._inductor.aoti_load_package("packaged_model.pt2")
out = loaded(x)
```

No JIT, no autotune, no recompile path. The shapes you serve must fall inside the `Dim` ranges you exported.

## When you do not want AOTInductor

- Training: AOTInductor is inference-only as of 2.8. There is no exported backward.
- Code paths with structural variability (different module branches at runtime). Export needs one static graph.
- Shape distributions outside what you exported. Either widen `Dim` ranges (compiles more variants into the archive) or accept that out-of-range shapes fall back.

## Reference

- [AOTInductor docs](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html).
- [Tutorial: torch.export + AOTInductor Python runtime](https://docs.pytorch.org/tutorials/recipes/torch_export_aoti_python.html).
- [vLLM low-hanging cold-start RFC #20451](https://github.com/vllm-project/vllm/issues/20451) — the production motivation.

## Common pitfalls

- **Export failed with "Dynamic shape error".** Either declare the dim with `Dim` or the example trace used a shape that's not representative. Pick example inputs that exercise the variability.
- **The `.pt2` is enormous.** Weights are bundled by default. To externalize: `aoti_compile_and_package(..., inductor_configs={"aot_inductor.package_constants_in_so": False})` (the exact option name has shifted across versions; check your `aoti_compile_and_package` signature).
- **Loaded model gives different numbers than eager.** Match dtype, device, and inference mode. AOTInductor expects the inputs you exported with.
