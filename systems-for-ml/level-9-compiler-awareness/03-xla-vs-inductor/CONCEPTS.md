# 03 — XLA vs Inductor

Two ML compiler stacks have momentum in 2026. They solve the same problem with different tradeoffs. Knowing which one a project lives in tells you most of what its performance ceiling and portability story look like.

## The two stacks side by side

```
PyTorch / Inductor stack                  JAX / XLA stack
────────────────────────                  ───────────────────
nn.Module                                 jit'd Python function
    │                                          │
    ▼  Dynamo (bytecode trace)                 ▼  jax.jit (abstract trace)
FX graph                                   jaxpr
    │                                          │
    ▼  AOTAutograd (decompose, functionalize)  ▼  lower
Post-grad ATen graph                       StableHLO
    │                                          │
    ▼  Inductor (fuse, schedule)               ▼  PJRT -> XLA compiler
Inductor IR                                HLO (XLA's IR)
    │                                          │
    ▼  codegen                                 ▼  target backend
Triton (GPU) / C++ (CPU)                   PTX / TPU IR / LLVM (CPU) / ROCm
    │                                          │
    ▼                                          ▼
PTX -> SASS                                hardware
```

Same chain shape, very different ecosystems and design choices.

## What each is optimized for

### Inductor
- **Single backend mindset, multiple codegens.** GPU-first via Triton, CPU via C++/OpenMP. The Triton output is readable and hackable.
- **Tight coupling to PyTorch eager.** Inductor lives in the PyTorch repo and the same eager-mode ATen ops show up in compiled graphs. No translation layer to a "neutral" IR.
- **Aggressive autotuning.** `max-autotune` mode tries multiple Triton configs per kernel and times them on the actual GPU.
- **JIT-friendly.** Designed to compile per-call, cache by guards. AOTInductor exists but is the second-class citizen.

### XLA
- **Multi-target by design.** Same StableHLO module compiles to GPU, TPU, CPU, and (via PJRT plugins) third-party hardware.
- **Whole-program optimization.** XLA traditionally fuses across the entire jit'd function. The fusion budget is bigger than Inductor's; the cost is longer compile times and less locality of debugging.
- **Strong AOT story.** StableHLO is versioned and serializable. You can compile today, deploy next year, on a different machine.
- **TPU-native.** TPUs have effectively only this stack. PyTorch on TPU goes through `torch_xla`, which lowers PyTorch to StableHLO so it can ride the same pipeline.

References:
- TorchInductor design — https://dev-discuss.pytorch.org/t/torchinductor-update-1/440
- StableHLO spec — https://openxla.org/stablehlo/spec
- OpenXLA project — https://openxla.org/
- PJRT (XLA's pluggable runtime) — https://openxla.org/xla/pjrt

## OpenXLA — the 2024–2026 split from Google

Until 2023, XLA was a Google project. In 2024 it moved to the OpenXLA umbrella with NVIDIA, AMD, Intel, AWS, Apple, and others as contributors. By 2026 the practical effect is:

- **StableHLO is the contract.** Frontends (JAX, PyTorch via torch_xla, TF Lite, Flax) emit StableHLO. Backends (XLA-GPU, XLA-TPU, IREE, third-party plugins) consume it.
- **NVIDIA invested in XLA-GPU.** The XLA GPU backend is no longer "TPU's poor cousin" — it has a real Triton emitter, cuDNN/cuBLAS integration, and competitive numbers on H100/B200 for many models.
- **Plugins via PJRT.** A new accelerator vendor implements a PJRT plugin, then any frontend that emits StableHLO can target that hardware. This is the de facto multi-backend ML compiler interface in 2026.

References:
- OpenXLA org — https://github.com/openxla
- PJRT plugins — https://openxla.org/xla/pjrt_integration

## Inductor vs XLA on three concrete dimensions

### Fusion granularity

```
Inductor:   one kernel ~= one elementwise chain (+ optional matmul epilogue)
XLA:        whole jit'd function, one big fused kernel where possible
```

Inductor's smaller fusion units make for shorter compile times and easier debugging — you can read each kernel and tell what it does. XLA's larger fusion units extract more performance on long pipelines but the resulting kernel is less readable and the failure modes (missed fusion, layout mismatch) take longer to diagnose.

### Shape policy

```
Inductor:   recompile per shape by default; mark_dynamic for shape-polymorphic graphs
XLA:        recompile per shape by default; jax.jit(..., static_argnums=...) and shape polymorphism via export
```

Both stacks suffer if you feed unbounded shape variation. JAX has a more mature shape-polymorphism story for AOT export (`jax.export`); PyTorch's equivalent (`torch.export` + dynamic dim hints) caught up by 2025 but is still less polished for serving-shaped workloads.

### Runtime story

```
Inductor:   PyTorch eager handles everything else; compiled regions plug in
XLA:        XLA owns the whole runtime; calls into cuDNN/cuBLAS/cuFFT as needed
```

This is why mixing `torch.compile`'d code with eager PyTorch is trivial (regions interleave) and why mixing JAX `jit` and non-jit code requires extra thought (the runtime boundary is sharper).

## When you'd see each in the wild

| Workload | Stack |
|---|---|
| Production LLM serving on NVIDIA | PyTorch + vLLM (Inductor) |
| Frontier-scale training on TPU | JAX + XLA |
| Frontier-scale training on NVIDIA | PyTorch + Megatron-Core / FSDP2 (Inductor optional) |
| Mobile / edge from a JAX model | JAX -> StableHLO -> IREE |
| Apple Silicon serving | PyTorch eager + MLX, or llama.cpp Metal — neither stack dominates |
| Custom accelerator (Groq, Cerebras, Tenstorrent) | StableHLO ingest, vendor-specific lowering |

The TPU answer is forced by hardware; the NVIDIA answer is forced by ecosystem (vLLM, FlashInfer, FlashAttention live in PyTorch land); the cross-platform answer is StableHLO + IREE (Topic 06).

## PyTorch on TPU — the bridge

`torch_xla` makes PyTorch run on TPUs by lowering ATen ops to StableHLO and dispatching through the XLA runtime. In 2026 it is production-ready for training but still has rough edges around dynamic shapes and certain custom ops. Google's internal flagships moved to JAX; PyTorch/XLA is for shops that can't or don't want to migrate frontends.

References:
- torch_xla — https://github.com/pytorch/xla
- PyTorch/XLA SPMD — https://pytorch.org/blog/pytorch-xla-spmd/

## "Why didn't one stack win"

Because the requirements pull in different directions:

- **Eager interop** is essential for PyTorch's research workflow. Inductor preserves it. XLA, by design, does not.
- **Multi-target portability** is essential for TPU shops and for non-NVIDIA hardware vendors. XLA/StableHLO provides it. Inductor doesn't (Triton is portable to AMD via ROCm but the broader story is GPU-only).
- **Compile time and iteration speed** matter for research. Inductor's smaller fusion units win here.
- **Deployment-time predictability** matters for production. StableHLO + AOT wins here.

So both stacks coexist and increasingly converge through MLIR — Triton's IR is MLIR-based, StableHLO is MLIR-based. The substrate is the same. The frontends and runtime models differ.

## What to walk away with

- Both stacks lower a high-level Python program to hardware code via progressive IRs. The shapes of the chains are nearly identical.
- The differences that matter day-to-day: fusion granularity, shape policy, runtime ownership, and which backends the stack targets.
- StableHLO is the multi-backend contract for the JAX/OpenXLA world. Inductor doesn't have a true equivalent — its "contract" is FX + ATen, which is PyTorch-internal.
- TPU work is XLA-shaped almost everywhere. NVIDIA serving is Inductor-shaped almost everywhere. Both can be wrong for your specific case; verify before you commit.
