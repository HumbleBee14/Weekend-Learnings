# Level 9 — Compiler Stack Awareness (High-Level Tour)

> Outer reference: [`systems-for-ml/README.md`](../README.md) · No project — reading + small writeup only
>
> Textbook companion: [Reddi Vol 1 — *AI Acceleration*](https://mlsysbook.ai/) for the hardware-side framing. Kiely doesn't cover compilers. If this level sparks interest, the sibling [`compiler-and-kernels/`](../../compiler-and-kernels/) track is the dedicated specialization path (Triton deep-dive, CUTLASS/CuTe-DSL, MLIR in practice, StableHLO/XLA).

## How to study this level

```
  Day 0 (15m)  ──►  Read the Scope note below ── this is an awareness tour
  Day 1-2 (3h) ──►  Reddi *AI Acceleration* chapter
                  + skim "What I Talk About When I Talk About IRs" (Lattner)
                  + skim the MLIR project paper (Lattner et al.)
  Day 2 → 5    ──►  Topics 01 → 08, mostly read-only. For each topic:
                       1. Open the topic folder's  README.md
                       2. Read its  CONCEPTS.md
                       3. Trace one model through torch.compile  (Topic 02)
  Day 6-7      ──►  Optional: write up one short post (~1 page) tracing one
                    model through the lowering stack. No project repo.
```

**Reference order when you get stuck:**
1. The topic's own `CONCEPTS.md`
2. Reddi *AI Acceleration*
3. PyTorch Dynamo / Inductor source for the torch.compile path
4. MLIR project docs for the IR side

**If you want to go deeper:** the sibling [`compiler-and-kernels/`](../../compiler-and-kernels/) track is months of dedicated study (Triton deep-dive, CUTLASS/CuTe-DSL, MLIR in practice, StableHLO/XLA, AI-assisted kernels). Don't try to cram it here.

**Compute:** Reading and small experiments. CPU is fine.

## Scope note (read this first)

By the time you're here, you've built an ML system end-to-end — train, optimize, serve, scale, ship, on cloud and on-device.

This week is **awareness, not specialization.** It exists so that:

- When someone mentions "MLIR / LLVM," you know what the line of work actually is.
- When `torch.compile` does something surprising, you can read the trace and roughly follow it down to hardware.
- When a colleague says "we're contributing kernels back to FlashInfer," you can hold the conversation.
- You can decide, with eyes open, whether the **AI Compiler Engineer** track is something you want to specialize in next — *separately from this curriculum*.

If you fall in love with this material, the deep-dive is months of dedicated study (LLVM Kaleidoscope tutorial → MLIR Toy → IREE/Triton/StableHLO source → contributing patches). That's a whole separate journey. This week is the tour.

## Where this fits

- **Comes after:** Levels 1–8. You should already understand `torch.compile`, FlashAttention, kernel fusion, and the GPU memory hierarchy — those are the user-level views of what compilers are doing underneath.
- **Comes before:** Nothing in this curriculum. This is the closing week. After this you build, explore, or specialize further.
- **Project this feeds:** None. Deliverable is a short writeup tracing one PyTorch model through `torch.compile` and Inductor.

## 2026 reality check — why this layer matters more than ever

- **Every fast inference engine is a compiler product.** vLLM uses `torch.compile` + CUDA graphs as its kernel-fusion layer. SGLang has its own scheduler-aware compile path. TensorRT-LLM is built on TensorRT (NVIDIA's compiler). FlashInfer dispatches to compiled kernels. None of these are hand-tuned C++ in 2026.
- **Custom-silicon companies are building fast.** Groq, Cerebras, Tenstorrent, Modular, MatX, Etched — each needs a compiler that targets their hardware. AI Compiler Engineering is one of the fastest-growing specializations in the field.
- **NVIDIA's CUTLASS 4.x** introduced **CuTe DSL** — a Python-fronted CUDA tensor-layout compiler used in production for "speed of light" GEMMs.
- **AI-assisted kernel synthesis** (AutoKernel, KernelBench, LLM-driven kernel generation) is a 2025–2026 research direction showing real wins. Reading-only awareness is enough this week.
- **MLIR has won as the multi-target ML IR.** PyTorch's Inductor lowers to Triton (which uses LLVM). JAX/XLA targets StableHLO (an MLIR dialect). IREE compiles MLIR to Vulkan/Metal/CUDA/CPU. Modular's Mojo is built on MLIR.

## Topic-by-topic deep dive

| # | Topic | What you'll be able to explain |
|---|-------|-------------------------------|
| 01 | the-lowering-picture | PyTorch graph → IR → hardware code, who does what |
| 02 | torch-compile-internals | Dynamo + Inductor, how Week 4's `torch.compile` actually works |
| 03 | xla-vs-inductor | The two big compilation paths and where each is used |
| 04 | what-mlir-and-llvm-are | Why an extra IR layer exists, what dialects mean |
| 05 | accelerator-landscape | How Groq / Cerebras / TPU / Tenstorrent compilers differ from GPU compilers |
| 06 | iree-and-portable-deployment | Compile once, run on Vulkan/Metal/CUDA/CPU |
| 07 | cutlass-and-cute-dsl | NVIDIA's tensor-layout compiler |
| 08 | ai-assisted-kernels | AutoKernel, KernelBench, LLM-driven kernel synthesis |

### 01 — `the-lowering-picture`

**The high-level view.**

```
PyTorch nn.Module
       │  (Dynamo: bytecode → FX graph)
       ▼
FX Graph IR  (Python-level, op-level)
       │  (Inductor: pattern-match + lower)
       ▼
Triton kernels  (Python DSL)
       │  (Triton compiler: LLVM IR backend)
       ▼
PTX  (NVIDIA assembly)
       │  (ptxas)
       ▼
SASS  (GPU machine code)
       │  (driver: load + launch)
       ▼
Hardware (SMs, warps, tensor cores)
```

Each arrow is an *optimization opportunity*. Each box is a different tool maintained by a different team. Each is also a place where things can go wrong (graph break, lowering failure, suboptimal kernel choice).

**The same picture for JAX.**

```
JAX program
       │  (jit + tracing)
       ▼
StableHLO  (MLIR dialect, vendor-neutral)
       │  (XLA compiler)
       ▼
HLO  (XLA's internal IR)
       │
       ▼ (target-specific lowering)
   ├── PTX (GPU)
   └── TPU instructions
```

**The unifier.** MLIR is the multi-level IR framework. Both stacks increasingly use MLIR as the connective tissue. StableHLO is an MLIR dialect. Triton's IR is MLIR-based. IREE is an MLIR-native compiler.

### 02 — `torch-compile-internals`

**Two phases.**

**Phase 1 — Dynamo.** Reads your Python bytecode (yes, the actual interpreter bytecode, not source). Builds an FX graph by symbolically executing the bytecode. Where it can't (dynamic control flow, custom ops without meta kernels, in-place mutations on views), it inserts a "graph break" — falls back to eager for that segment, resumes graph capture after.

**Phase 2 — Inductor.** Takes the FX graph. Pattern-matches: fusable elementwise chains, matmul + bias + activation, attention shapes. Lowers to:
- **Triton kernels** for GPU. Inductor *generates* Triton code from the graph.
- **C++/OpenMP** for CPU.
- **CUDA graphs** for stable steady-state — captures the launch sequence, replays it.

**Why graph breaks matter so much.** A break splits your model into compiled regions with eager code in between. Each break re-pays kernel launch overhead, breaks fusion across the boundary, and prevents CUDA graph capture. Engineers who understand this can read `TORCH_LOGS=graph_breaks` output and know what to fix.

**Build steps (minimal).**
1. Take a tiny model (3-layer transformer block).
2. Wrap with `torch.compile`. Run with `TORCH_LOGS="graph_breaks,recompiles,output_code"`.
3. Read the generated Triton code. It's surprisingly readable.
4. Manually introduce a graph break (e.g., `print(x.shape)` mid-forward). See the output change.

### 03 — `xla-vs-inductor`

**XLA / StableHLO.** Google's path. Used by JAX, by TensorFlow (legacy), by PyTorch/XLA on TPUs. StableHLO is the portable IR — vendor-neutral, designed to be a stable contract between frontend frameworks and hardware backends.

**Inductor.** PyTorch's path. Triton-first, GPU-first. Closer to the metal but less portable.

**When you'd see each.**
- Working at Google or on TPUs: XLA / StableHLO.
- Working in PyTorch on NVIDIA / AMD GPUs: Inductor.
- Wanting one IR to target many backends: StableHLO via IREE.

### 04 — `what-mlir-and-llvm-are`

**LLVM.** Compiler infrastructure project from the 2000s. IR (LLVM IR) is *flat* — single set of instructions, single optimization pipeline. Every modern compiler (Clang, Rust, Swift) lowers through LLVM IR eventually.

**The problem ML hit.** ML programs have *high-level* operations (matmul, conv, attention) that benefit from optimization at that level (tile sizes, fusion patterns). LLVM IR is too low-level to express "this is a matmul that should be tiled differently on H100 vs A100."

**MLIR.** Multi-Level IR. Solves this by allowing *multiple* IRs ("dialects") to coexist in the same compiler, with progressive lowering between them. Standard dialects:
- **`linalg`** — generic tensor operations.
- **`affine`** — loop-nest representation.
- **`gpu`** — GPU launch primitives.
- **`tosa`** — Tensor Operator Set Architecture (vendor-neutral ML ops).
- **`stablehlo`** — Google's portable ML IR.
- **`triton`** — Triton's IR.

A model gets lowered: `pytorch ops → linalg → affine → gpu → llvm`. Each step is a separate optimization opportunity.

**When does this matter to you.** When `torch.compile` does something surprising or when you want to write a kernel that targets multiple hardware backends. Otherwise, MLIR is invisible — but it's the substrate underneath.

### 05 — `accelerator-landscape`

**GPU compilers (NVIDIA / AMD).** Target a SIMT execution model with deep memory hierarchy. Optimizations: tiling for shared memory, warp specialization, async DMA. Tools: CUDA, Triton, CUTLASS.

**TPU compilers (Google).** Target a systolic array. Optimizations: matrix-shape alignment, pipeline scheduling, hbm/vmem placement. Tool: XLA / StableHLO.

**Groq compilers.** Target a deterministic dataflow architecture (no caches, no schedulers — the compiler statically schedules every instruction). Optimizations: instruction-level scheduling, memory placement at compile time.

**Cerebras compilers.** Target a wafer-scale chip with 850K cores. Optimizations: layer-pipeline scheduling across the wafer, weight streaming.

**Tenstorrent compilers.** Target Tensix cores with explicit programming model. TT-Metal (low-level), TT-Buda / TT-NN (higher-level).

**Modular.** Mojo language + MAX engine. Compiler-first ML stack. Targets multiple accelerators from one source.

**The pattern.** Each new accelerator company is, in large part, a compiler company. The hardware is half the product; the compiler that gets models running fast is the other half.

### 06 — `iree-and-portable-deployment`

**IREE.** MLIR-native compiler. Compile once → run on Vulkan, Metal, CUDA, ROCm, CPU. The portable-deployment story.

**When you'd reach for it.** When you have one model that needs to run on heterogeneous hardware (some CUDA, some Apple Silicon, some Android). Or when you want vendor-neutrality.

**Light-touch reading.** [iree.dev](https://iree.dev/). Skim the architecture page. Know it exists.

### 07 — `cutlass-and-cute-dsl`

**CUTLASS.** NVIDIA's CUDA Templates for Linear Algebra Subroutines. C++ template library that lets you write custom GEMM kernels with the same performance as cuBLAS.

**CuTe DSL** (CUTLASS 4.x, 2025–2026). A Python-fronted DSL for tensor layouts. Replaces some of the C++ template wizardry with Python-level abstractions while compiling to the same fast PTX.

**When you'd see it.** Custom kernel work for FlashInfer, vLLM, TRT-LLM. Speed-of-light GEMMs where cuBLAS isn't the right fit.

**Light-touch.** Read the CUTLASS README. Look at one CuTe DSL example. You don't write CUTLASS this week — but you should know what it is.

### 08 — `ai-assisted-kernels`

**The frontier.** LLMs writing kernels.

- **AutoKernel** — LLM-driven kernel generation, claims faster kernels than human-written for some shapes.
- **KernelBench** — benchmark for evaluating LLM kernel generation.
- **Sakana AI's CUDA agent** — wrote competitive matmul kernels via iterative LLM generation.

**Status in 2026.** Research, not production. The trajectory is clear: standard, well-shaped kernel patterns are increasingly automatable. The remaining human work is novel hardware targeting, novel attention variants, and the architectural decisions about *which* kernels to write.

## Output for this week

Not a project repo. Just one document: `reports/compiler-tour.md`.

**Structure (≤2000 words).**

1. **The lowering picture** for one model. Take Qwen2.5-0.5B (small, traceable). Run `torch.compile` with `TORCH_LOGS=output_code`. Capture the generated Triton code for one transformer block. Annotate it: "this is the matmul, this is the softmax, this is the layernorm, here's where they're fused."
2. **Where graph breaks happened** in your runs (if any). What caused them, what fixing them would buy.
3. **MLIR / LLVM in plain language.** A two-paragraph summary you'd give a teammate who asked "what is MLIR and why do I keep seeing it?"
4. **Accelerator landscape.** A table comparing GPU / TPU / Groq / Cerebras / Tenstorrent at the *compiler* level. What execution model, what optimizations, what tools.
5. **Decision.** One paragraph: "based on what I learned this week, here's whether I want to specialize in compiler engineering next, and here's my next step if so."

That's the artifact. No graphs, no benchmarks, no failure injection. Reading and synthesis.

## Definition of done

- [ ] You ran `torch.compile` with output-code logging and read the generated Triton.
- [ ] You can sketch the lowering picture (PyTorch → FX → Triton → PTX → SASS) on a whiteboard.
- [ ] You can explain MLIR and why it exists in two paragraphs.
- [ ] You can name three non-NVIDIA accelerator companies and what their compiler stacks are.
- [ ] You wrote `reports/compiler-tour.md`.
- [ ] You made a deliberate decision about whether to specialize further or move on.

## Resources

- **PyTorch — `torch.compile` deep dive** — [pytorch.org/blog](https://pytorch.org/blog/) (search "Inductor"). The 2024 "How `torch.compile` works" post is the best plain-English explanation.
- **MLIR primer** — [mlir.llvm.org](https://mlir.llvm.org/). "Why MLIR?" page.
- **MLIR Toy tutorial** — [mlir.llvm.org/docs/Tutorials/Toy](https://mlir.llvm.org/docs/Tutorials/Toy/). The canonical "build a small compiler in MLIR" walkthrough. Read, don't necessarily implement.
- **LLVM Kaleidoscope tutorial** — [llvm.org/docs/tutorial](https://llvm.org/docs/tutorial/). Build a tiny language frontend. Read, optionally implement.
- **StableHLO** — [openxla.org/stablehlo](https://openxla.org/stablehlo).
- **IREE** — [iree.dev](https://iree.dev/).
- **Triton** — [triton-lang.org](https://triton-lang.org/).
- **CUTLASS** — [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass). README + one CuTe example.
- **Modular Mojo** — [modular.com/mojo](https://www.modular.com/mojo).
- **Sakana AI's CUDA agent post** — for the AI-assisted-kernels frontier (find via blog search).

## Common pitfalls

1. **Trying to learn this deeply in one week.** Please avoid that — the point of this week is awareness. Compiler engineering is a deep subject in its own right, and if you find yourself wanting to write a custom MLIR pass, that's a great signal to come back to it as a separate, dedicated track later.
2. **Skipping `TORCH_LOGS=output_code`.** Reading the generated Triton is the single highest-leverage exercise. It demystifies `torch.compile` faster than any blog post.
3. **Treating MLIR as "another LLVM."** It's not. The "multi-level" part is the whole point. Single-IR mental models will mislead you.
4. **Believing AI-generated kernels are production-ready.** As of 2026 they're a research area with promising results, not a deployment path.

## What you'll be able to do after this week

> Read `torch.compile` / Inductor traces and explain how a PyTorch graph lowers through Dynamo → FX → Triton → PTX. Understand the MLIR/LLVM ecosystem at a conceptual level — StableHLO, IREE, Triton's IR, dialect-based progressive lowering. Describe compiler differences across GPU / TPU / Groq / Cerebras / Tenstorrent.

This is a tour, not a deep specialization — and that's intentional. Compiler engineering as a specialty is a separate, deeper track.

**If you want to go deeper:** months of dedicated study. Roadmap: LLVM Kaleidoscope (1 week) → MLIR Toy (1 week) → read StableHLO source (2 weeks) → contribute a small patch to Triton or IREE (months). Companies doing deep compiler work: NVIDIA, Apple, AMD, Groq, Cerebras, Tenstorrent, Modular.
