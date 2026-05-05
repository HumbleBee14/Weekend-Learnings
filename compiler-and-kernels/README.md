# compiler-and-kernels — The Layer Between Your Code and the Hardware

**Prerequisite:** Complete `systems-for-ml/` Levels 1–9 first. You need to have built an inference server, profiled it, optimized it, benchmarked real engines, and deployed a platform. This track takes that foundation and teaches you what's happening *inside* the tools you've been using.

## What this track is

`systems-for-ml` taught you to use the kernel layer. This track teaches you to *read, write, and extend* it.

Every tool you used in `systems-for-ml` has a kernel layer underneath it:
- `torch.compile` → Dynamo traces bytecode, Inductor emits Triton kernels
- FlashAttention → tiled Triton kernels with online softmax
- vLLM's MoE dispatch → fused Triton experts kernel
- TensorRT-LLM → CUTLASS C++ / CuTe-DSL GEMM templates
- Tenstorrent TT-Forge → MLIR dialect lowering chain
- JAX on TPU → StableHLO → XLA GPU backend

After this track you can open any of those and follow what's happening. More importantly, you can *contribute to them* — write a new fused kernel, fix a graph break, add a custom attention variant, port an op to a new hardware backend.

## The progression

```
systems-for-ml/          you used Triton kernels
  ↓
Level 1  Triton deep     you write advanced Triton kernels (persistent, warp-specialized)
Level 2  torch.compile   you trace Dynamo bytecode, fix graph breaks, read Inductor output
Level 3  FlashAttention  you understand FA2→FA3→FA4 tiling; write custom attention in FlexAttention
Level 4  CuTe DSL        you write GEMM kernels in CuTe-DSL for Hopper/Blackwell
Level 5  Kernel fusion   you diagnose when to fuse, write the fused op, measure the win
Level 6  MLIR practical  you write an MLIR optimization pass; follow TT-Forge/IREE lowering
Level 7  StableHLO/XLA   you use StableHLO as a portability layer; run on TPU and CPU
Level 8  AI kernels      you use AutoKernel + KernelBench; evaluate and review AI-generated code
Level 9  Rust for infra  you write a Rust-based tokenizer + router; understand Candle/mistral.rs
```

## Why this order

Triton first because it's the most direct extension of what you already did in `systems-for-ml` Level 2. You know what a GPU kernel does — now you write production-quality ones. `torch.compile` second because once you can write Triton you can read what Inductor emits, which makes the Dynamo/Inductor internals concrete rather than abstract. FlashAttention and CuTe-DSL deepen the two specific kernel families that dominate LLM inference. Kernel fusion synthesizes those skills. MLIR is the compiler substrate underneath all of it — you study it after you know what it's compiling. StableHLO is the portability story that lets the kernel work travel across hardware. AI-assisted kernels is the frontier. Rust is the infrastructure layer that glues everything into a production system.

## What you'll be able to do after this track

- Write production Triton kernels with warp specialization and TMA; benchmark against Liger-Kernel
- Read `torch.compile` output, diagnose graph breaks with `depyf`, implement the vLLM piecewise CUDA graph pattern
- Write custom attention variants (sliding window, ALiBi, custom masks) using FlexAttention's `score_mod`/`mask_mod`; understand FA2→FA3→FA4 tiling progression
- Write a GEMM kernel in CuTe-DSL for Hopper (SM90) and Blackwell (SM100); benchmark against cuBLAS
- Profile a transformer's kernel fusion landscape; write hand-fused Triton ops that beat `torch.compile` on bandwidth-bound ops
- Write an MLIR optimization pass; follow TT-Forge's lowering from PyTorch through TTIR/TTNN to Metalium
- Export a JAX model to StableHLO; deploy via IREE to CPU and Metal (your M5 Mac)
- Run AutoKernel on your own model overnight; critically evaluate the generated kernels
- Write a Rust-based tokenizer + routing layer; understand where Candle/mistral.rs sit vs Python inference

## Compute needed

| Level | Hardware | Cost estimate |
|---|---|---|
| 1 — Triton deep | Colab T4 (free) or RunPod T4 ($0.40/hr) | < $5 |
| 2 — torch.compile | Colab T4 / any GPU | < $5 |
| 3 — FlashAttention | RunPod A100 ($2/hr) | ~$10 |
| 4 — CuTe DSL | RunPod A100 for SM90; H100 optional ($3/hr) | ~$15 |
| 5 — Kernel fusion | RunPod A100 | ~$10 |
| 6 — MLIR | M5 Mac (CPU) + IREE Metal | $0 |
| 7 — StableHLO/XLA | Google Colab TPU (free) | $0 |
| 8 — AI kernels | RunPod A100 (overnight run) | ~$15 |
| 9 — Rust infra | M5 Mac or any CPU | $0 |

**Total cloud spend: ~$60 for the full track.** Most levels run on free Colab or your Mac.

## Projects

| Level | Project artifact |
|---|---|
| 1 | Fused RMSNorm+RoPE Triton kernel with warp specialization; benchmark vs Liger-Kernel |
| 2 | Graph-break audit of a LLaMA block; piecewise CUDA graph pattern implementation |
| 3 | Sliding-window + ALiBi attention via FlexAttention; FlashInfer ragged batching demo |
| 4 | BF16 persistent GEMM in CuTe-DSL on SM90; NVFP4 variant on SM100 |
| 5 | Kernel fusion profiling report: top-3 bandwidth-bound ops, hand-fused Triton vs compile |
| 6 | Out-of-tree MLIR pass: tile a `linalg.matmul`; run through IREE CPU + Metal |
| 7 | Flax model → StableHLO export → IREE GPU + CPU benchmark |
| 8 | AutoKernel overnight run; kernel quality review report |
| 9 | Rust tokenizer + prefix-routing layer; P50/P99 latency vs Python equivalent |

## Resources (outer level — each inner README has topic-specific ones)

- **Triton** — [triton-lang.org](https://triton-lang.org/)
- **Liger-Kernel** — [github.com/linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- **depyf** — [github.com/thuml/depyf](https://github.com/thuml/depyf)
- **FlashAttention-4** — [arxiv.org/abs/2603.05451](https://arxiv.org/abs/2603.05451)
- **FlexAttention** — [pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention/)
- **FlashInfer** — [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **CUTLASS / CuTe-DSL** — [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- **Colfax Research CUTLASS tutorials** — [research.colfax-intl.com](https://research.colfax-intl.com/)
- **MLIR** — [mlir.llvm.org](https://mlir.llvm.org/)
- **Tenstorrent tt-mlir** — [github.com/tenstorrent/tt-mlir](https://github.com/tenstorrent/tt-mlir)
- **IREE** — [iree.dev](https://iree.dev/)
- **StableHLO** — [openxla.org/stablehlo](https://openxla.org/stablehlo)
- **KernelBench** — [github.com/ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- **AutoKernel** — [github.com/RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel)
- **HuggingFace Candle** — [github.com/huggingface/candle](https://github.com/huggingface/candle)
- **mistral.rs** — [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- **vLLM Router (Rust)** — [blog.vllm.ai/2026/01/05/vllm-sr-iris.html](https://blog.vllm.ai/2026/01/05/vllm-sr-iris.html)
