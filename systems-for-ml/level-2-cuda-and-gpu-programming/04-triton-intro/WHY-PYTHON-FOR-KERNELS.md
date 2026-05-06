# Why kernel work moved to Python in 2026

A note on the language choice. Worth reading before going further.

## The shift

Up to roughly 2023, the answer to "I need a custom GPU kernel" was: write CUDA C++. By 2026, the default answer is: write Triton (Python). The frontier (FlashAttention-4, NVFP4 GEMMs) has moved one step further to CuTe-DSL, which is also Python.

This is not because C++ got worse. It's because the costs and benefits shifted.

## What Python (Triton/CuTe-DSL) gives you

**1. Iteration speed.** A Triton kernel reload is sub-second. A CUDA C++ kernel rebuild is 10–60 seconds. Multiply that by hundreds of iterations during tuning and the gap dominates how much you can experiment.

**2. The autotuner finds better configs than you do.** `@triton.autotune` benchmarks 5–50 candidate configurations per shape. Humans trying to hand-pick `BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages` for every shape lose to the autotuner consistently. The autotuner doesn't get tired or biased.

**3. Hardware portability with one source.** Same Triton `.py` runs on:
- NVIDIA T4 (sm_75) — uses HMMA tensor cores
- NVIDIA A100 (sm_80) — uses 3rd-gen tensor cores
- NVIDIA H100 (sm_90) — uses WGMMA
- NVIDIA B200 (sm_100) — uses tcgen05 + TMEM
- AMD MI300X — uses HIP/ROCm tensor cores

The compiler emits the right instruction for each. CUDA C++ requires you to write specialized templates per architecture.

**4. The compiler handles the hard parts.** Bank conflicts (swizzling), register allocation, warp scheduling, double buffering, async copies — all things you'd hand-code in CUDA C++. In Triton you write tile-level operations and the compiler decides.

**5. Triton 3.2's automatic warp specialization.** Producer warps (TMA loads) and consumer warps (WGMMA compute) are partitioned by the compiler. Hand-writing this in CUDA C++ is multi-week work; in Triton you get it for free if your kernel matches the pattern.

**6. Native PyTorch integration.** A Triton kernel is just a Python function. You call it like any other op. CUDA C++ extensions need a bridge (PyBind11, `cpp_extension`).

## What CUDA C++ still gives you

**1. Reading existing code.** vLLM `csrc/`, PyTorch internals, cuDNN, cuBLAS, NCCL — all CUDA C++. The serving stack you'll work with has thousands of lines of it. You need to read it. Reading != writing, but the syntax is the same.

**2. Custom logic that doesn't fit Triton's tile model.** Some kernels need irregular memory access patterns or fine-grained control over individual threads (rare in LLM serving, common in graph algorithms or sparse work).

**3. Embedded / edge inference.** llama.cpp's CUDA backend, TensorRT custom layers, on-device GPU compute on phones. Python isn't available there.

**4. Maximum performance for niche shapes.** A specialist with a month and a deep-cut workload can beat Triton via raw CUDA + PTX inline assembly. This is a small population of people doing a small fraction of total kernel work.

## Concrete examples — what does each company actually use?

| Project | Primary kernel language | Why |
|---|---|---|
| vLLM PagedAttention (`csrc/`) | CUDA C++ | Existing code from 2023; would likely be Triton if rewritten today. They are migrating piece by piece. |
| vLLM new MoE kernels | Triton | New work, all Triton |
| vLLM Triton attention backend | Triton | New backend launched 2026, intended to eventually replace the C++ one |
| SGLang attention kernels | Triton + some CUDA | Triton dominant; CUDA for very specific paths |
| FlashInfer | Mix of CUDA C++ + Triton + CuTe-DSL | The dispatcher uses Triton for JIT compilation per dtype/shape; some legacy paths in C++ |
| Liger-Kernel (LinkedIn) | Triton entirely | New 2024-2026 project, all Triton |
| FlashAttention-2 | CUDA C++ | Tri Dao wrote it in 2023 |
| FlashAttention-3 | CUDA C++ | Hopper-specific, hand-tuned |
| **FlashAttention-4** | **CuTe-DSL (Python)** | **New 2026 — even Tri Dao moved to Python** |
| NVIDIA CUTLASS | C++ templates | Library, not kernels |
| NVIDIA CuTe-DSL | Python | Where new GEMM frontier work is happening |
| cuDNN, cuBLAS internals | CUDA C++ | NVIDIA-internal, not user-extensible |
| vLLM Router (gateway) | Rust | Not a kernel — but shows the broader shift toward better tools for each layer |

The pattern: **new kernel work is Python; old kernels stay C++ until rewritten**. The trajectory is clear.

## What this means for what you learn

- **Read CUDA C++** — needed to navigate existing codebases. Topics 2 and 3 of this level are correct.
- **Write Triton** — the production language for new kernels. Topic 4.
- **Read CuTe-DSL** — for understanding where the GEMM/attention frontier is going. Covered in `compiler-and-kernels` Level 4.
- **Don't try to write WGMMA + TMA + warp-specialized matmul in CUDA C++** — that's what CUTLASS exists for. Even at frontier labs, individuals don't hand-write that anymore.

## The Tri Dao FA2 → FA4 progression as evidence

The clearest evidence: the most-cited GPU kernel author of the last 5 years.

- **FA2 (2023)** — CUDA C++ in `dao-ailab/flash-attention/csrc/`. Thousands of lines of templated C++.
- **FA3 (2024)** — CUDA C++, more templates, even more hand-tuning for Hopper.
- **FA4 (2026)** — CuTe-DSL, **Python**. Tri Dao chose this language for the new frontier kernel.

If the author of FlashAttention writes new kernels in Python, that's the field's answer.

## So when do you reach for CUDA C++?

In 2026, my honest answer:

1. **You're modifying existing CUDA C++ code** (vLLM, PyTorch, etc.). Read it, edit it, don't rewrite it.
2. **You're targeting a non-NVIDIA, non-AMD, non-Apple device** with a custom toolchain.
3. **You're doing kernel research** at a frontier lab and have a month to hand-tune a single op.
4. **You're working on edge / embedded** where Python is unavailable.

Otherwise: Triton, every time.

## References

- Tri Dao on writing FA4 in CuTe-DSL — https://tridao.me/blog/2026/flash4/
- Liger-Kernel paper (Triton-only production kernels) — https://arxiv.org/abs/2410.10989
- The Anatomy of a Triton Attention Kernel (105% of SOTA, Triton only) — https://arxiv.org/abs/2511.11581
- vLLM Triton attention backend (Mar 2026) — https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html
