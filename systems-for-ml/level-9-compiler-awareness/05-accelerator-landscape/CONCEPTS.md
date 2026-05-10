# 05 — Accelerator Landscape (the compiler view)

Every non-NVIDIA accelerator company is, in large part, a compiler company. The silicon is half the product; the toolchain that turns PyTorch graphs into fast schedules is the other half. This topic is the map: what each architecture looks like, what its compiler has to solve, what the entry-point tooling is in 2026.

## Why this is a compiler problem, not a kernel problem

GPUs got away with hand-written kernels for a long time because the SIMT model is forgiving — warps execute, caches absorb mispredictions, the hardware schedules around stalls. Most other accelerators are *less forgiving*:

- Systolic arrays need the matmul shape aligned to the array dimensions or you waste MACs.
- Dataflow architectures (Groq, Cerebras) have **no runtime scheduler at all** — every instruction is placed at compile time.
- Wafer-scale and multi-chiplet parts need explicit pipeline scheduling across hundreds of cores.

The closer you get to "the silicon does exactly what the compiler said, no more," the more the compiler is the product.

## The landscape, 2026

```
                           Execution model           Primary toolchain (2026)
   ─────────────────────────────────────────────────────────────────────────
   NVIDIA  Blackwell B200/B300   SIMT + tensor cores     CUDA, Triton, CUTLASS 4.x / CuTe DSL
   AMD     MI300X / MI350X       SIMT + matrix cores     ROCm, Triton (AMD backend), Composable Kernel
   Google  TPU v5p / v6 / v7     Systolic + VPU          XLA / StableHLO / Pallas
   AWS     Trainium2             Systolic + scalar       Neuron SDK (XLA-based) + NKI kernel DSL
   Groq    LPU                   Static dataflow         GroqWare compiler (closed)
   Cerebras CS-3 / CS-4          Wafer-scale dataflow    Cerebras SDK, weight streaming
   Tenstorrent Wormhole/Blackhole Tensix cores            TT-Metal (low) / TT-NN / TT-Buda
   SambaNova SN40L               Reconfigurable dataflow SambaFlow (MLIR-based)
   Modular  -                    Targets all of above    Mojo + MAX engine
```

State as of early 2026: NVIDIA still dominates training; for inference, the alternatives are real and shipping (Groq for low-latency text, Cerebras for fast inference of large models, Trainium2 for cost on AWS, MI300X/MI350X for memory-bound workloads).

## SIMT GPUs (NVIDIA, AMD)

Single Instruction Multiple Thread. A warp/wavefront of 32/64 threads executes the same instruction. Memory hierarchy is deep: registers → shared memory / LDS → L2 → HBM.

What the compiler has to solve:

- **Tiling** — pick block sizes that fit shared memory and saturate tensor cores.
- **Async pipelining** — overlap HBM loads (`cp.async`, `cp.async.bulk` on Hopper+) with compute.
- **Warp specialization** — split a CTA into producer/consumer warps so loads and MMAs run concurrently.
- **Register allocation** — the kernel is fast or slow based on register pressure and spilling.

Tooling layer cake:

```
PyTorch op
   │   torch.compile / Inductor
   ▼
Triton DSL (Python)                ← language layer
   │   Triton compiler (MLIR-based)
   ▼
LLVM IR
   │   NVPTX / AMDGPU backend
   ▼
PTX (NVIDIA) / GCN ISA (AMD)
   │   ptxas / amdgcn assembler
   ▼
SASS / native ISA
```

CUTLASS sits parallel to Triton — also targets PTX, but via C++ templates (and now CuTe DSL, see Topic 07). AMD's analog is Composable Kernel (CK) plus a Triton AMD backend that has matured significantly through 2025.

References:
- Triton language — https://triton-lang.org/main/index.html
- Hopper warp specialization patterns — https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
- AMD ROCm Triton backend — https://github.com/ROCm/triton

## Systolic arrays (TPU, Trainium2)

A grid of MAC units that pumps data through in lockstep. A 128x128 systolic array does a 128x128 matmul tile per cycle once filled. Dataflows:

```
   weights stream down ↓        outputs accumulate diagonally
   inputs stream right →

   [W][W][W][W]
   [W][W][W][W]    Each cell: multiply incoming x and w,
   [W][W][W][W]    add to partial sum from above, pass right.
   [W][W][W][W]
```

What the compiler has to solve:

- **Shape alignment** — pad/tile every matmul to the array's native dimensions or the unused MACs are dead silicon.
- **HBM ↔ on-chip memory placement** — TPUs have VMEM (vector memory) and HBM; XLA decides when to spill/reload.
- **Pipeline scheduling** — overlap weight load, compute, and output drain.

Toolchain:

```
JAX program / TF / PyTorch-XLA
   │   tracing
   ▼
StableHLO (MLIR)
   │   XLA compiler
   ▼
HLO IR
   │   target-specific lowering
   ▼
TPU ISA (Google) / Neuron ISA (AWS)
```

For TPUs, **Pallas** (https://docs.jax.dev/en/latest/pallas/index.html) is the kernel-author DSL — Triton-like, but emits MLIR that XLA can still optimize around. Used for FlashAttention-on-TPU. AWS Neuron has **NKI** (Neuron Kernel Interface) playing the same role on Trainium2.

References:
- StableHLO — https://openxla.org/stablehlo
- Pallas — https://docs.jax.dev/en/latest/pallas/index.html
- AWS Neuron NKI — https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/

## Static dataflow (Groq LPU)

No caches. No branch predictors. No runtime scheduler. Every functional unit's activity for every cycle is determined at compile time. The compiler emits a literal cycle-by-cycle schedule for the whole chip.

What this buys: deterministic latency. The same input always takes the same number of cycles. Token-generation latency on Groq is famously stable.

What it costs: the compiler is the entire product. Any model that runs is a model the compiler has been taught to schedule. Adding new ops or new attention variants is compiler work, not kernel work.

```
Model graph
   │   Groq compiler (closed)
   ▼
Per-cycle instruction trace
   for every functional unit on every chip in the system
```

Reference (architecture overview): https://groq.com/wp-content/uploads/2024/07/GroqRack-Compute-Cluster-Datasheet.pdf

## Wafer-scale (Cerebras)

CS-3: ~900,000 cores on one wafer-scale die, 44 GB of on-chip SRAM. CS-4 announced 2025. Programming model: **weight streaming** — model weights live in external memory (MemoryX), stream into the wafer per layer; activations live on-chip.

What the compiler has to solve:

- **Layer-pipeline placement** — assign which layer runs on which region of the wafer.
- **Streaming schedule** — load layer N's weights while computing layer N-1.
- **Sparse compute** — Cerebras hardware natively supports unstructured sparsity, the compiler needs to exploit it.

Cerebras has been the surprise inference winner of 2025 for large open-weight models — Llama-405B, DeepSeek-V3-class — because the entire model's working set fits on-chip and there's no HBM bottleneck per token.

Reference: https://cerebras.ai/blog (architecture posts).

## Tensix cores (Tenstorrent)

A grid of small RISC-V cores, each with a matrix engine and explicit local memory. Programming is *closer to the metal* than CUDA — the user (or the compiler) writes data-movement programs explicitly.

Stack:

```
Model
   │
   ├── TT-Buda    (high level, automatic placement)
   │
   ├── TT-NN      (PyTorch-like op library on Metal)
   │
   └── TT-Metal   (low-level, explicit kernel + data movement)
```

TT-Metal is open source (https://github.com/tenstorrent/tt-metal). Tenstorrent is one of the few non-NVIDIA companies betting on an open ecosystem; their bet is that compiler engineers will contribute kernels back the way Triton has gathered a community.

## SambaNova, Etched, MatX

Briefer notes:

- **SambaNova SN40L** — reconfigurable dataflow, MLIR-based compiler. Targets enterprise inference of large models.
- **Etched Sohu** — transformer-only ASIC. Compiler is narrower (only has to handle attention + MLP + a few activations) which lets them push throughput higher at the cost of any model that doesn't fit the transformer mold.
- **MatX** — training-focused ASIC, early.

The pattern: the more the silicon specializes, the simpler the compiler can be, but the narrower the workload it accepts.

## Modular as the cross-cutting bet

Modular's pitch: write once in **Mojo**, compile to all of the above. Mojo is a Python-superset language built on MLIR, designed to expose hardware-level performance knobs (SIMD width, memory layout, parallelism) without dropping into C++.

The MAX engine is the runtime that ships compiled artifacts to the right backend. Whether this becomes the unifying layer or stays one of several is the open question; through 2025 Modular gained real production users for Llama-class inference on heterogeneous fleets.

References:
- Mojo — https://www.modular.com/mojo
- MAX — https://www.modular.com/max

## The pattern to take away

For each accelerator, ask the same five questions:

1. What is the execution model? (SIMT, systolic, dataflow, wafer)
2. Where does the user write kernels, if at all? (DSL, op library, none)
3. What IR does the high-level frontend lower to? (Triton, StableHLO, custom)
4. Who controls scheduling — runtime or compiler?
5. What is the on-chip memory hierarchy, and who places data in it?

Answer those and you can roughly predict where the compiler engineering work happens, what the per-vendor moat looks like, and why the same model has very different perf characteristics on different silicon.

## What's actually changing in 2026

- **Trainium2 is shipping at scale** on AWS, with Neuron SDK + NKI mature enough for production inference of large open models.
- **TPU v6 (Trillium)** is in broad availability; v7 announced 2025 with significant FP8/FP4 changes — XLA changes track this.
- **MI350X (CDNA4)** brings AMD into FP4 territory; ROCm 7 + Triton AMD backend close most of the SW gap that existed in 2023.
- **CUTLASS 4.x + CuTe DSL** is the production way to write custom NVIDIA kernels (see Topic 07).
- **Cerebras inference** went from "demo" in 2024 to "real provider" in 2025 — the compiler did most of that work.
- **Modular MAX** is the only credible "compile once, run on many accelerators" production stack; IREE (Topic 06) is the open-source counterpart.
