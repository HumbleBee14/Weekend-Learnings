# 02 — Dynamo bytecode and FX graphs, with depyf

You write a 10-line model, dump everything Dynamo and Inductor produced for it with [depyf](https://github.com/thuml/depyf), and read the output by hand. By the end of this sub-module the words "Dynamo transformed bytecode" and "FX graph" mean specific files on your disk, not abstractions.

## Hardware

CPU works. A GPU is fine too. The output is more interesting on GPU (you see the Triton kernel as well as the FX graph) but the Dynamo and FX layers are GPU-independent.

## What to run

```bash
pip install torch>=2.8 depyf
python tiny_model_dump.py
```

This produces a `./dump/` directory. Open it.

Then add the deliberate graph break (uncomment the marked line in [`tiny_model_dump.py`](tiny_model_dump.py)) and re-run with `--out dump_broken`. Compare.

## What you should observe

In `dump/`:

- `full_code_for_forward_<id>.py` — the original Python source plus the dispatcher that decides which compiled variant to call.
- `__transformed_code_for_forward.py` — the bytecode Dynamo wrote, decompiled back to Python. **Read this.** The top is a block of guards (`___check_type`, shape checks); below is the call into the compiled FX graph.
- `__compiled_fn_<n>.py` — for each FX graph: the graph nodes (one per tensor op), and then the Inductor-lowered Python (if GPU, includes Triton kernels).

In `dump_broken/`:

- Two `__transformed_code_for_*.py` files instead of one — Dynamo cut the function into two compiled regions around the `print`.
- Two `__compiled_fn_*.py` files.
- The dispatcher in `full_code_for_*` is more complex, with calls to both pieces.

Write three things in [`notes.md`](notes.md):
1. The full guard list from the un-broken version. Note which guards are about types, which are about shapes, which are about other attributes.
2. The FX graph node list. Map each node back to a line in your Python source.
3. The diff between un-broken and broken: how many compiled regions, how the dispatcher changed.

## Reference reading

- [depyf walk-through](https://depyf.readthedocs.io/en/latest/walk_through.html) — the canonical example.
- [Introducing depyf — PyTorch blog](https://pytorch.org/blog/introducing-depyf/).
- [Torch.compile can be debugged now! — dev-discuss](https://dev-discuss.pytorch.org/t/torch-compile-can-be-debugged-now/1595).

## Common pitfalls

- **You ran `prepare_debug` but the dump dir is empty.** The context manager needs to wrap the *call site*, not the definition. See the file.
- **You don't see Triton in `__compiled_fn_*.py`.** You're on CPU. That's expected; the C++ codegen appears in `output_code.py` instead.
- **The transformed code looks identical to your source.** Look more carefully — the guards at the top, the names like `___check_type` are the giveaway. Dynamo's job for simple code *is* to mostly leave it alone.
