# Level 2 — Learning Path

The goal of this level is **kernel literacy, not kernel mastery**. After this you can read kernels in vLLM, SGLang, and FlashInfer and follow what they're doing. Writing your own production kernel is its own track (`compiler-and-kernels`).

## Read first

- **`LANGUAGE-CHOICES.md`** — why this level mixes CUDA C++ and Python. Worth 5 minutes before diving in. Explains what an industry stack actually looks like in 2026.

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-cuda-mental-model/` | 1–2h | Grid → block → warp → thread, drawn from memory. Memory hierarchy preview. SIMT understood. |
| `02-first-cuda-kernels/` | 2–3h | Three working kernels (vector add, vec4 ReLU, softmax) in **CUDA C++** + a PyTorch inline version. Understands coalescing, reductions, online softmax. |
| `03-matrix-multiply/` | 3–4h | Three matmul kernels (naive, coalesced, SMEM-tiled) in **CUDA C++**, benchmarked against cuBLAS. Boehm's first 3 steps. |
| `04-triton-intro/` | 2–3h | Vector add, matmul, fused softmax×y in **Triton (Python)**. Plus `WHY-PYTHON-FOR-KERNELS.md` explaining the language shift. |
| `05-gpu-memory-hierarchy/` | 2–3h | Bandwidth measurement at each level. Plus **`READING-PRODUCTION-KERNELS.md`** — read PagedAttention CUDA C++ + Liger-Kernel Triton + vLLM Triton attention. The most industry-shaped exercise here. |
| `06-flash-attention-walkthrough/` | 3–4h | Online softmax in NumPy + minimal FA2 in Triton + **`READING-FLASH-ATTENTION.md`** — read dao-ailab's CUDA C++ FA2, FlashInfer's Python dispatcher, vLLM's Triton FA, and FA4 in CuTe-DSL. The 200-word writeup test. |

## Each topic folder

- `CONCEPTS.md` — the theory, with diagrams, real-2026 numbers, and reference URLs
- One or more code files (`.cu` or `.py`) with comments explaining each piece
- `README.md` — quickstart commands, expected output, things to try

Some topics also have:
- `GPU-NOTES.md` (Topic 1) — pocket reference for GPU concepts that come up

## What hardware you need

- **Minimum**: a free Colab T4 (sm_75). Works for everything except topic-3's "compare to cuBLAS at scale" runs which look slow on T4.
- **Recommended**: rent an A100 hour ($1-2/hr on RunPod or Vast) for the matmul and FA topics. The numbers are more meaningful and FA2 actually shines.
- **For the FA3/FA4 reading**: no hardware needed. You're reading papers, not running their code.

## What you should NOT try this level

- Writing FA3 or FA4 yourself
- Writing a production-quality WGMMA + TMA matmul in raw CUDA C++
- Reproducing cuBLAS performance on a Hopper or Blackwell GPU in raw CUDA

These are weeks of work even for experts. Read the worklogs (Pranjal Shankhdhar, Hamza Elshafie, Modular's Blackwell series) for understanding. Implementation belongs in the `compiler-and-kernels` track.

## What goes in your reports

Level 2 doesn't have its own G1/G2-style required graphs (those start in Level 3 with profiling). But three artifacts to keep:

1. **Boehm-step matmul plot.** TFLOPS at each step (naive, coalesced, SMEM-tiled, cuBLAS). Carry to Level 3 to overlay roofline analysis.
2. **Triton matmul vs CUDA matmul comparison.** Same problem, two implementations. Note the line counts.
3. **The 200-word FlashAttention writeup.** This is the artifact that tests whether you got Topic 6.

## After this level

Level 3 introduces real profiling tools: Nsight Systems for timeline traces, Nsight Compute for per-kernel deep dive, the roofline model. You'll profile `mini-serve` from Level 1 and find the bottleneck that motivates Level 4's paged KV cache work.

Level 2 was about *understanding what kernels do*. Level 3 is about *measuring why they're slow*. Level 4 is about *fixing the slow ones*. The chain is intentional.
