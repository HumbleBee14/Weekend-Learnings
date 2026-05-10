# 02 — torch.compile Internals

Level 4 Topic 07 covered `torch.compile` as a *user* of inference compilation — piecewise CUDA graphs, vLLM defaults, gotchas. This topic is one layer underneath: what Dynamo and Inductor actually are, what each pass does, what the artifacts on disk mean. The deep version is `compiler-and-kernels/level-2-torch-compile-internals/`. This is the awareness map.

## The two halves

```
                   torch.compile
                   ─────────────
            ┌───────────────────────────┐
            │                           │
            ▼                           ▼
        Dynamo                      Inductor
   (frontend / capture)        (backend / codegen)
            │                           │
   Python bytecode -> FX        FX -> Triton/C++
   Guards, graph breaks         Fusion, scheduling, codegen
   torch.fx.GraphModule         Triton source, .cubin, .so
```

Dynamo's job: turn a Python callable into a graph. Inductor's job: turn the graph into fast kernels. They communicate by passing FX graphs (and a small protocol around guards).

## Dynamo — what's actually happening

Dynamo is a **bytecode-level tracer**. It does not parse your Python source. It hooks into CPython's frame evaluation API (PEP 523) and runs your function's bytecode through a symbolic interpreter.

```
def forward(self, x):
   y = self.up(x)    →  CALL_METHOD bytecode
   y = self.act(y)   →  CALL_METHOD bytecode
   return self.down(y)
```

The symbolic interpreter walks the bytecode op by op:

- For each tensor op, it records an FX node.
- For each control-flow op (`POP_JUMP_IF_FALSE`, etc.), it inspects whether the condition depends on a tensor's *value* (untraceable) vs its *metadata* (traceable as a guard).
- For each Python construct it can't model (printing, dict mutation in some cases, custom C extensions), it emits a **graph break**: it ends the current FX graph, lets eager Python run the offending bit, then starts a new graph after.

The output is one or more FX `GraphModule`s plus a set of **guards** — runtime checks that say "this graph is valid as long as `x.shape == [4, 64]`, `x.dtype == fp32`, no autograd state changed, etc." On the next call, Dynamo evaluates the guards; if they pass, it skips tracing and reuses the compiled artifact.

Why guards matter: every cache miss is a recompile. With wide shape variation (typical for LLM inference) and no shape policy, you can spend more time recompiling than running.

References:
- Dynamo overview — https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html
- PEP 523 (the frame eval hook Dynamo uses) — https://peps.python.org/pep-0523/
- Guards deep dive — https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-dynamic-shapes-work/925

## AOTAutograd — the bridge

Between Dynamo and Inductor sits AOTAutograd. It does three things:

1. **Joint trace** — runs autograd against the FX graph to produce a single forward+backward graph. (For inference, just the forward.)
2. **Decompose** — rewrites composite ATen ops into simpler "core" ops. `nn.functional.silu` becomes `x * sigmoid(x)`. `layer_norm` becomes a sequence of mean/var/normalize/affine.
3. **Functionalize** — removes in-place mutations and aliasing so the graph is pure. `x.add_(y)` becomes `x_out = x + y` and downstream uses get rewired to `x_out`.

Why this matters: Inductor wants a *flat, pure, low-level* graph because that's what fusion and scheduling are easiest on. AOTAutograd is the conversion.

## Inductor — what's actually happening

Inductor is a graph compiler. It runs roughly these phases:

```
  Post-grad ATen graph
        │
        │  pattern_matcher
        ▼  (constant folding, layout propagation,
        │   matmul/conv pattern matches, attention rewrites)
        │
  "Lowered" graph of Inductor IR nodes
        │
        │  scheduler
        ▼  (group nodes into kernels, pick fusion boundaries,
        │   choose loop ordering, pick block sizes)
        │
  Scheduler "buf" nodes per kernel
        │
        │  codegen
        ▼  (Triton for GPU, C++/OpenMP for CPU)
        │
  Generated source -> compiler -> launchable kernel
```

The fusion decision is the load-bearing one. Inductor groups elementwise chains into a single kernel by default, fuses elementwise into the epilogue of a matmul where the matmul template supports it, and lets attention through as a "ExternKernel" call to FlashAttention/FlashInfer rather than codegen-ing it from scratch.

`max-autotune` mode adds: try multiple Triton configs (block sizes, num_warps, num_stages) per kernel, time them on the actual GPU, pick the winner. This is where the additional 10–30% comes from on tight kernels, paid for in compile time.

References:
- TorchInductor design — https://dev-discuss.pytorch.org/t/torchinductor-update-1/440
- Inductor codegen overview — https://docs.pytorch.org/docs/stable/torch.compiler_inductor_profiling.html
- max-autotune — https://docs.pytorch.org/docs/stable/torch.compiler_inductor.html

## Reading the artifacts

```bash
TORCH_LOGS="output_code" python your_model.py 2> trace.log
TORCH_COMPILE_DEBUG=1 python your_model.py    # dumps to ./torch_compile_debug/
```

What you'll find:

- `output_code` — the actual generated Triton (or C++). Read this. It's surprisingly clean.
- `torch_compile_debug/run_*/` — per-compilation directory:
  - `fx_graph_readable.py` — the FX graph as code.
  - `fx_graph_runnable.py` — the same graph wrapped in a runnable script.
  - `output_code.py` — generated kernel + wrapper.
  - `graph_diagram.svg` — visualization.

For diagnosing graph breaks at the bytecode level, **depyf** decompiles the Dynamo-produced bytecode back to readable Python: https://github.com/thuml/depyf — covered in the deep track.

## Graph breaks — what they actually cost

A break splits one compiled region into two. Concretely:

```
[ compiled region 1 ]
        │
        │  Python call boundary
        │  (eager runs here, including the break-causing op)
        │
[ compiled region 2 ]
```

Each boundary costs:

- One Python-level call (microseconds).
- Lost fusion across the boundary — intermediate tensors must be materialized to HBM and re-read.
- No CUDA graph capture across the boundary — every region needs its own capture, every replay has its own launch overhead.
- Potential extra recompilations if the input shapes to region 2 vary independently.

`TORCH_LOGS=graph_breaks` prints the location and reason for every break. The reason text is usually specific enough to act on ("data-dependent control flow on tensor value", "untraceable C function call", etc.).

Common avoidable causes:

- `print()` or logging mid-forward.
- `.item()` or `.tolist()` to extract a scalar before using it in Python control flow.
- Custom ops without a registered "meta" kernel (Dynamo can't infer output shapes).
- Iterating over a tensor (`for row in x`).

## The 2026 production pattern: piecewise CUDA graphs

Covered in Level 4 Topic 07. Recap from the compiler angle:

- Most of the model has **static shapes** per token (the matmuls in projections and MLP).
- Attention has **dynamic shape** (KV cache length grows).
- `torch.compile` + CUDA graph capture works great on static-shape regions.
- vLLM V1 deliberately splits the forward into pieces: pre-attention (compiled+captured), attention (eager → FlashAttention/FlashInfer kernel), post-attention (compiled+captured).

This is a *cooperative* design between the inference engine and the compiler. You don't get it from `torch.compile` alone.

References:
- vLLM piecewise CUDA graph blog (Aug 2025) — https://blog.vllm.ai/2025/08/20/torch-compile.html
- vLLM torch.compile design doc — https://docs.vllm.ai/en/latest/design/v1/torch_compile.html

## What to walk away with

- Dynamo traces *bytecode*, not source. That's why it can handle most Python.
- AOTAutograd sits between Dynamo and Inductor and does the unglamorous flattening.
- Inductor's win is **fusion** plus **autotuning** plus **codegen quality**. All three matter.
- Graph breaks have a specific, reasoned cost — kernel launch overhead, lost fusion, broken CUDA graph capture.
- The production recipe in 2026 is not "compile the whole forward" — it's piecewise compile + CUDA graph capture around the attention kernel.
