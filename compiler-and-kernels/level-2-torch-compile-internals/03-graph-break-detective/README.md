# 03 — Graph-break detective

Six minimal snippets, each broken differently. For each: diagnose with `_dynamo.explain`, fix, verify with `fullgraph=True`. By the end you will recognize every common pattern at sight.

## What a graph break actually does

```
   forward(x):                            Dynamo's view:
   ─────────────                          ──────────────
     y = linear(x)        ──►   ┌──────────────────────────┐
     y = norm(y)                │  Compiled region 1 (FX)  │
     y = act(y)                 │   linear → norm → act    │
                                └─────────────┬────────────┘
                                              ▼
     print(y.sum())       ──►       [ EAGER  PYTHON ]
                                    unsupported op runs
                                    as plain CPython
                                              ▼
     z = linear2(y)       ──►   ┌──────────────────────────┐
     z = act2(z)                │  Compiled region 2 (FX)  │
     return z                   │   linear2 → act2         │
                                └──────────────────────────┘
```

*One Python function, two FX graphs, one eager island in between. Every break costs you a kernel-launch boundary and (if guarded badly) a recompile.*

## Hardware

CPU works. The breaks are independent of backend.

## The drill, per snippet

For each file `break_NN_*.py`:

1. Run it: `python break_NN_*.py`. It will print Dynamo's explanation and the first failure.
2. Read the explanation. Identify the cause from the level README's list of six.
3. Edit the file's `forward()` (or its caller) to apply the fix. Each file has a `# FIX HERE` marker.
4. Re-run with the env var `FULLGRAPH=1` to verify no graph break remains:
   ```
   FULLGRAPH=1 python break_NN_*.py
   ```
5. Write your diagnosis in [`notes.md`](notes.md): cause, fix applied, whether the fix is free or has a measurable cost.

## The six breaks

| # | File | Pattern |
|---|---|---|
| 01 | [`break_01_item_in_conditional.py`](break_01_item_in_conditional.py) | `.item()` inside `if`. |
| 02 | [`break_02_print_in_forward.py`](break_02_print_in_forward.py) | `print(...)` mid-forward. |
| 03 | [`break_03_data_dependent_if.py`](break_03_data_dependent_if.py) | `if tensor.sum() > 0:`. |
| 04 | [`break_04_optional_kwargs.py`](break_04_optional_kwargs.py) | Optional `**kwargs` plumbing (the HF Transformers pattern). |
| 05 | [`break_05_tolist_numpy.py`](break_05_tolist_numpy.py) | `.tolist()` + numpy round-trip. |
| 06 | [`break_06_custom_class.py`](break_06_custom_class.py) | A custom Python class flowing through forward. |

## Reference

- [Common Graph Breaks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html).
- [Use fullgraph=True to identify breaks](https://docs.pytorch.org/docs/stable/compile/programming_model.fullgraph_true.html).
- [GraphMend (arXiv 2509.16248)](https://arxiv.org/abs/2509.16248) — for the catalog of auto-fixable patterns. Don't run it; read the patterns section.

## Common pitfalls

- **You "fixed" a break by setting `fullgraph=False`.** That's not a fix, that's hiding the symptom. The break still exists and the compile still skips that region. Use `fullgraph=True` as the diagnostic.
- **You used `torch.cond` for everything.** `torch.cond` has overhead and a narrow contract (both branches must produce same-shape outputs). Use it when the break is genuine data-dependent control flow, not when a simple lift-out works.
- **You set `capture_scalar_outputs = True` and called it a day.** That option captures `.item()` symbolically — it can be the right fix, but you should know it can also make downstream guards weirder. Read the value, decide deliberately.
