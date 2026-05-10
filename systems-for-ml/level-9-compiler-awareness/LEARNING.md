# Level 9 — Learning Path

The closing level. Awareness, not specialization. Eight topics that map the compiler stack underneath every fast inference engine and every non-NVIDIA accelerator.

```
The lowering picture          (01)        the spine: PyTorch -> IR -> hardware
torch.compile internals       (02)        Dynamo, AOTAutograd, Inductor
XLA vs Inductor               (03)        the two big production paths
MLIR and LLVM                 (04)        the substrate everything sits on
Accelerator landscape         (05)        TPU, Trainium, Groq, Cerebras, Tenstorrent
IREE                          (06)        compile once, many backends (open source)
CUTLASS and CuTe DSL          (07)        speed-of-light NVIDIA kernels
AI-assisted kernels           (08)        the frontier: LLMs writing Triton/CUDA
```

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-the-lowering-picture/` | 1-2h | A whiteboard sketch from PyTorch nn.Module down to SASS, naming each tool. |
| `02-torch-compile-internals/` | 2-3h | Dynamo as bytecode tracer, AOTAutograd as flattener, Inductor as fuser+codegen. Reading `TORCH_LOGS=output_code`. |
| `03-xla-vs-inductor/` | 1-2h | When each path is used, why JAX and PyTorch settled on different stacks, where StableHLO unifies them. |
| `04-what-mlir-and-llvm-are/` | 1-2h | LLVM as the flat-IR endpoint, MLIR as the multi-level IR framework, dialects as the unit of abstraction. |
| `05-accelerator-landscape/` | 2h | Five-question framework, per-vendor toolchain, why each new accelerator company is a compiler company. |
| `06-iree-and-portable-deployment/` | 1-2h | StableHLO -> Linalg -> Flow -> Stream -> HAL, why the runtime is small, where IREE fits. |
| `07-cutlass-and-cute-dsl/` | 2-3h | CUTLASS as kernel-author library, CuTe as layout algebra, CuTe DSL (Python) as the 2025-2026 entry point. |
| `08-ai-assisted-kernels/` | 1-2h | KernelBench, Sakana's CUDA agent, what works and what doesn't in 2026. |

Total: ~12-18 hours of focused reading and inspection. Notably shorter than other levels — this is a tour, not a build.

## What's new in 2026 (deltas vs 2024 content)

- **Inductor + piecewise CUDA graphs** has settled as the canonical inference compile recipe — vLLM V1, SGLang, TRT-LLM all use variants of it.
- **MLIR has won as the multi-target ML IR** — Triton, IREE, StableHLO, Modular Mojo all sit on it. Single-IR mental models from the LLVM era are misleading.
- **CUTLASS 4.x + CuTe DSL** replaced template-heavy CUTLASS for new kernel work. FlashAttention 3/4, FlashInfer, DeepGEMM, TRT-LLM all use it.
- **Trainium2 went mainstream** on AWS in 2025; Neuron SDK + NKI is real production tooling.
- **Cerebras inference for large models** is genuinely competitive with GPU racks for token throughput on Llama-405B-class workloads.
- **Modular MAX engine** is the production "compile once, many backends" closed-source play; IREE is the open-source counterpart.
- **AI-assisted kernels** moved from research curiosity to "useful for autotuning and first-draft kernels," with frontier models now writing correct (if not always fast) Triton on first attempt.
- **TPU v6 (Trillium)** generally available; v7 announced 2025.
- **AMD MI350X (CDNA4)** brought FP4 to AMD; ROCm 7 + Triton AMD backend closed most of the SW gap.
- **DeepSeek's DeepGEMM and FlashMLA** showed open-source community can ship CUTLASS-grade kernels independently of NVIDIA.

## What hardware you need

- **None for awareness.** Every topic is doable on a laptop. The IREE export, the CUTLASS reading, the KernelBench walk, the accelerator-landscape script — all CPU-only or just-reading.
- **A GPU helps for one specific exercise** — running `torch.compile` with `TORCH_LOGS=output_code` on a real GPU and reading the generated Triton (Topic 02). CPU shows the C++ codegen path, which is also informative.
- **No exotic hardware required.** TPU / Trainium / Groq / Cerebras are read-only — you're learning what their compilers do, not running them.

## Each topic folder

Same shape as the rest of the curriculum:

- `CONCEPTS.md` — theory, 2026 state, ASCII diagrams, inline references.
- One or more code/inspection files (small Python scripts that dump IRs, walkthroughs that point at real repo files, shell snippets showing toolchain flow).
- `README.md` — quickstart, what to look for, things to try, where it goes next.

## Output for this level

Not a project repo. One document: `reports/compiler-tour.md` (the level README has the structure — five sections, ~2000 words, the lowering picture for one model + MLIR explanation + accelerator table + the specialization decision).

## After this level

Two paths.

**If you fall in love with this material:** specialize. The roadmap is months, not weeks.

```
LLVM Kaleidoscope tutorial      (1 week)
MLIR Toy tutorial               (1 week)
Read StableHLO source           (2 weeks)
Read one Triton MLIR pass       (1 week)
Contribute small patch to IREE / Triton  (months)
```

Companies doing deep compiler work in 2026: NVIDIA, Apple, AMD, Google, Groq, Cerebras, Tenstorrent, Modular, AWS Annapurna, Meta (PyTorch core).

**If this confirmed you don't want to specialize:** good. You now have:

- Enough to read `torch.compile` traces and explain Dynamo / Inductor.
- A whiteboard-ready picture of MLIR / LLVM and why both exist.
- The accelerator landscape map — what each vendor's compiler is called and where it sits.
- Calibrated expectations for AI-assisted kernel work.

That's the awareness this level was for. The rest of the curriculum (Levels 1-8) gave you the system; this level gave you the substrate underneath. Pick what you build next based on which layer pulled at you.
