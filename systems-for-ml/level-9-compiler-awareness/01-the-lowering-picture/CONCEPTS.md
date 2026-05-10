# 01 — The Lowering Picture

The whole point of this level: when something in `torch.compile` or vLLM goes sideways, you should be able to picture which box on the lowering chain is responsible. This topic draws the chain. Every later topic zooms into one box.

## The PyTorch chain (Inductor path)

```
nn.Module (Python)
        │  Dynamo: symbolic bytecode trace
        ▼
FX Graph IR              ← Python-level, op-level, easy to read
        │  AOTAutograd: forward + backward traced together,
        │  decomposed to ATen "core" ops, functionalized
        ▼
ATen / "post-grad" graph ← lower-level ops, no in-place mutation
        │  Inductor: pattern-match, fuse, schedule
        ▼
Inductor IR              ← scheduler nodes, loop ranges, layouts
        │  codegen
        ▼
   ┌────┴────┐
   ▼         ▼
Triton    C++/OpenMP     (GPU vs CPU backend)
   │         │
   │  Triton compiler: TTIR → TTGIR → LLVM IR (MLIR-based)
   ▼         ▼
PTX (NVIDIA)    object .so (CPU)
   │  ptxas
   ▼
SASS                     ← actual GPU machine code
   │  driver loads, kernel launches
   ▼
SM execution             ← warps, tensor cores, async copies
```

Five compilers in a trench coat. Each transition is a separate codebase, separate maintainers, separate failure modes.

References:
- PyTorch compiler architecture overview — https://docs.pytorch.org/docs/stable/torch.compiler_get_started.html
- Inductor design doc — https://dev-discuss.pytorch.org/t/torchinductor-update-1/440
- vLLM "How torch.compile works" Aug 2025 — https://blog.vllm.ai/2025/08/20/torch-compile.html

## The JAX chain (XLA / StableHLO path)

```
JAX program (Python)
        │  jax.jit: tracing on abstract values
        ▼
jaxpr                    ← JAX's small functional IR
        │  lower
        ▼
StableHLO                ← MLIR dialect, vendor-neutral, versioned
        │  PJRT / XLA compiler
        ▼
HLO                      ← XLA's internal IR (legacy, still alive)
        │  target lowering
        ▼
   ┌────┼─────┬──────────┐
   ▼    ▼     ▼          ▼
LLVM   PTX   TPU IR    ROCm
(CPU) (GPU) (XLA-TPU)  (AMD)
```

Same shape: progressively lower IR until you hit something the hardware can execute. The differences from the PyTorch chain matter:

- StableHLO is **versioned** with backwards/forwards compatibility guarantees. You can serialize a StableHLO module today and load it next year. FX graphs have no such promise.
- StableHLO targets **multiple backends from one IR**. Inductor is GPU-first via Triton; CPU is a separate codegen path.
- XLA does **whole-program optimization** by default (operator fusion across the entire jit'd function). Inductor does aggressive fusion too but the boundaries are different — Dynamo decides where graphs start and end.

References:
- StableHLO spec — https://openxla.org/stablehlo/spec
- XLA architecture — https://openxla.org/xla/architecture
- jaxpr docs — https://docs.jax.dev/en/latest/jaxpr.html

## What MLIR is doing in both pictures

By 2026 MLIR is the connective tissue:

```
                MLIR substrate
   ┌──────────────────────────────────────┐
   │  StableHLO    Triton-IR    linalg     │
   │  tosa         arith        gpu        │
   │  affine       memref       nvgpu      │
   │  vector       transform    llvm       │
   └──────────────────────────────────────┘
```

Each is a "dialect" — a self-contained set of operations and types — and any compiler built on MLIR can mix them. Triton's compiler runs `ttir → ttgir → llvm` passes. IREE runs `stablehlo → linalg → vector → spirv/llvm`. The same lowering machinery, the same pass infrastructure, different choices of dialects per project.

Topic 04 covers MLIR proper. The thing to lock in here: MLIR is *not* a competitor to LLVM. It sits *above* LLVM and produces LLVM IR (or PTX, or SPIR-V, or whatever) at the bottom.

## Where things go wrong on the chain

Each arrow is a place to lose performance or correctness:

| Boundary | Failure mode |
|---|---|
| Python → FX | Graph break (untraceable Python construct) |
| FX → ATen | Decomposition explodes a single op into many |
| ATen → Inductor | Fusion missed, redundant memory traffic |
| Inductor → Triton | Suboptimal tile size, low occupancy |
| Triton → PTX | LLVM IR pessimization, register spill |
| PTX → SASS | ptxas reorders, exposes a latent bug |
| Driver → SM | Launch overhead dominates at low batch |

When `torch.compile` is "slow" the cause is usually one or two of these, and `TORCH_LOGS` plus a profiler tells you which.

## What "lowering" actually means

Lowering = rewriting the program in a representation that's closer to the machine, optionally losing high-level structure that's no longer needed. A few invariants every level holds:

1. **Each level is an IR** — a typed, structured program representation, not a string.
2. **Lowering is monotonic** in detail — you don't go back up. (Some compilers cheat with "raise" passes, but it's the exception.)
3. **Optimization happens at every level**, not just one — you fuse at the FX/ATen level (high-level ops), tile at the linalg level (loops), schedule at the LLVM level (instructions).
4. **The lowest IR is hardware-specific** — PTX for NVIDIA, SPIR-V for Vulkan, TPU IR for TPUs. Above that everything is portable in principle.

## JIT vs AOT for ML

The "what is a compiler doing here" question splits two ways:

**JIT (Just-In-Time)**
- `torch.compile` defaults to JIT — first call triggers trace + compile.
- Pros: shape-specialized code, no separate build step, works in notebooks.
- Cons: cold-start latency (30–60s for 7B model), recompiles on shape change.

**AOT (Ahead-Of-Time)**
- `AOTInductor`, `torch.export`, StableHLO+IREE all produce a serialized artifact you can load and run.
- Pros: no cold start, deployable as a `.so` or `.vmfb`, deterministic.
- Cons: must commit to shapes (or pay for dynamic shape support), more friction in iteration.

In 2026 production serving, the answer is "JIT with caching" — vLLM compiles once, caches in `~/.cache/vllm/torch_compile_cache`, and subsequent restarts are fast. AOTInductor gets used for cold-start-sensitive deployments (serverless, edge).

References:
- AOTInductor — https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html
- torch.export — https://docs.pytorch.org/docs/stable/export.html
- IREE compile-then-run model — https://iree.dev/guides/

## What to walk away with

- The PyTorch and JAX chains as a single mental picture; both terminate in machine code.
- MLIR is the substrate underneath both, not a side track.
- Each arrow on the chain is a real codebase you could read, and a real failure mode you could hit.
- JIT vs AOT is a deployment question, not a fundamental property of the compiler.
