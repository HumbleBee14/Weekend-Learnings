# Why this level mixes CUDA C++ and Python

If you started reading this level expecting "CUDA = C++, top to bottom," some of the topics will look surprising. Here's the reasoning, with industry context.

## The 2026 reality

For new GPU kernel work, the field has moved to Python — Triton for general kernels, CuTe-DSL for the GEMM/attention frontier. CUDA C++ is still everywhere in *existing* codebases, and reading it is a real skill, but the *write* path is increasingly Python.

Strongest evidence: FlashAttention-4 (March 2026, Tri Dao) is written in **CuTe-DSL — Python**. The most-cited GPU kernel author of the last 5 years moved the new frontier kernel to Python. That's the field's answer.

## Per-topic language choices

| Topic | Primary language | Why |
|---|---|---|
| 01 — CUDA mental model | Notes + tiny demo | Conceptual; one Triton snippet just to show warp/block layout |
| 02 — first kernels | **CUDA C++** | Vector add, ReLU, softmax in raw CUDA — the right way to internalize threads, warps, shared memory, `__syncthreads()`. You can't learn these in Triton. |
| 03 — matmul | **CUDA C++** | Boehm's progression (naive → coalesced → SMEM-tiled). The *only* way to feel why each step wins is to write each one. |
| 04 — Triton intro | **Python (Triton)** | This is the production language. Same matmul as Topic 3 in 1/4 the lines, with autotune. |
| 05 — memory hierarchy | Python + reading C++ | Bandwidth measurement script in Triton (because the lesson is the *numbers*, not the language). Reading exercise on real vLLM + Liger-Kernel sources. |
| 06 — FlashAttention | Python + reading C++ | Minimal FA2 written in Triton (because that's how Tri Dao's tutorial does it now). Reading exercise on dao-ailab CUDA C++, FlashInfer, vLLM Triton FA, and FA4 in CuTe-DSL. |

## What an industry serving stack looks like in 2026

The 2026 inference serving stack mixes both:

```
Python layer:
  ┌───────────────────────────────────────────────┐
  │  Application / FastAPI / your business logic  │
  ├───────────────────────────────────────────────┤
  │  vLLM / SGLang / TRT-LLM Python interfaces    │
  ├───────────────────────────────────────────────┤
  │  Triton kernels (RMSNorm, RoPE, MoE, fused)   │
  │  FlashInfer Python dispatcher                 │
  └───────────────────────────────────────────────┘
                       ↓
C++ layer:
  ┌───────────────────────────────────────────────┐
  │  vLLM C++ extensions (PagedAttention)         │
  │  FlashInfer C++ kernels                       │
  │  CUTLASS templates                            │
  └───────────────────────────────────────────────┘
                       ↓
CUDA layer:
  ┌───────────────────────────────────────────────┐
  │  cuBLAS, cuDNN, NCCL (NVIDIA libraries)       │
  └───────────────────────────────────────────────┘
                       ↓
Hardware:
  ┌───────────────────────────────────────────────┐
  │  PTX → SASS → Tensor Cores → HBM              │
  └───────────────────────────────────────────────┘
```

Both languages live in the same project, with each layer doing what it's best at:

- **Python** for orchestration, dispatch, JIT compilation, autotune, integration with PyTorch
- **C++** for cross-platform serving (vLLM Worker), embedded use, existing kernel codebases
- **CUDA / CUTLASS / Triton** as the actual compute primitives (mix of C++ templates and Python DSLs)

## What the curriculum is teaching you

Two distinct skills:

1. **Read CUDA C++.** You'll touch existing codebases. vLLM, PyTorch, cuDNN, FlashAttention's CUDA path — all C++. Topics 2 and 3 build the muscle to read these. Topics 5 and 6's reading exercises put you directly in real code.

2. **Write Triton.** New kernel work. Topic 4 plus production examples in Topics 5 and 6.

Both skills matter. Most jobs that touch GPU kernels are 70% reading and 30% writing — and the writing increasingly happens in Python.

## What this curriculum doesn't teach

- **Hand-writing WGMMA + TMA + warp-specialized matmul in CUDA C++.** That's CUTLASS / CuTe-DSL territory in 2026. The `compiler-and-kernels` track (Level 4) covers CuTe-DSL. Even at frontier labs, individuals don't hand-write that anymore.
- **PTX / SASS inline assembly.** A small population of people write this for the absolute fastest kernels. Out of scope.
- **Driver / HAL / firmware development.** Embedded systems track, not ML systems.

## Recommended reading order in this level

1. Topics 1, 2, 3 — CUDA C++ to internalize the model
2. Topic 4's `WHY-PYTHON-FOR-KERNELS.md` — the language framing
3. Topic 4's code — Triton in action
4. Topic 5's `READING-PRODUCTION-KERNELS.md` — real vLLM + Liger-Kernel sources
5. Topic 6's algorithm + minimal Triton FA + `READING-FLASH-ATTENTION.md`

After all of that you'll have:
- Written 3–5 kernels in CUDA C++
- Written 3 kernels in Triton (matching or beating your CUDA versions)
- Read ~2000 lines of production CUDA C++ and Triton
- Felt the gap between hand-tuned C++ and Triton autotune
- Seen FA4 written in Python and understood why

## The honest summary

If a curriculum tells you "to do GPU work you must master CUDA C++," it's a 2018 curriculum. If it tells you "Python is enough," it's missing the reading skill that matters in real codebases.

The 2026 answer is both, with each used for what it's good at. That's what this level teaches.
