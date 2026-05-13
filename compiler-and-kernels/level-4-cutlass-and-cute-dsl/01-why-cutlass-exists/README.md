# 01 — Why CUTLASS exists

> Outer: [`../README.md`](../README.md) · Hardware: none

This submodule is read-and-think. No code. Its purpose is to make sure you have the right mental model before you write anything in CuTe-DSL, because the three layers (CUTLASS the library, CuTe the algebra, CuTe-DSL the Python frontend) are routinely conflated and the conflation produces bad kernels.

## The historical arc, briefly

**2017 — CUTLASS 1.0.** NVIDIA open-sourced a C++ template library that replicated cuBLAS's GEMM performance while letting users specialize the kernel — pick the tile shape, swap the epilogue, add a custom precision. cuBLAS is a closed binary; CUTLASS is header-only C++. Why this mattered: deep-learning research kept inventing new ops (mixed precision, sparse attention, quantization variants) that cuBLAS could not target on its release cadence.

**2020 — Ampere SM80, cp.async.** Hardware async copy from GMEM to SMEM landed. CUTLASS got a pipelined mainloop that overlapped copies with compute. The kernel-author's vocabulary added the word "stage."

**2022 — Hopper SM90, TMA + WGMMA + thread block clusters.** TMA replaced the warp-driven `cp.async` with a hardware descriptor that one instruction kicks off and the warp moves on. WGMMA replaced the per-warp MMA with a per-warpgroup async MMA. Thread block clusters let CTAs in the same cluster share SMEM via the new "distributed shared memory." CUTLASS 3.0 rebuilt around these — and around CuTe.

**2022–2024 — CuTe.** Cris Cecka et al. at NVIDIA built CuTe (Cute Template) as the algebra inside CUTLASS 3.x. Every tensor — in HBM, in SMEM, in registers, in the new TMEM on Blackwell — became a Layout `(shape, stride)`. Composition rules let you describe a TMA descriptor, a swizzled SMEM tile, and a register fragment in the same language. The point: tile-mapping decisions stopped being hidden in C++ templates and became expressions you could see and reason about.

**2024 — Blackwell SM100, tcgen05 + TMEM + NVFP4.** Tensor cores got bigger (`m128n256k16` MMA tile), got their own memory (TMEM), got their own MMA family (`tcgen05`), and got 4-bit floats with hardware block scaling. The MMA is now issued by a single thread on behalf of a CTA, and pairs of CTAs can cooperatively run one MMA. The kernel-author's vocabulary added "TMEM allocation," "CTA-pair MMA," and "block scale."

**May 2025 — CuTe-DSL.** NVIDIA shipped a Python frontend that uses CuTe directly. JIT-compiles Python → custom IR → MLIR → PTX. Compile times 20–30× faster than C++ templates; same hardware control; same kernel performance. The CUTLASS team called this "CUTLASS 4."

**Sep 2025 — FlashAttention-4 ships in CuTe-DSL.** Tri Dao's team chose CuTe-DSL over Triton and over C++ CUTLASS. The Modal reverse-engineering writeup is the proof point: five specialized warp types, TMEM accumulator, `tcgen05.mma.cta_group::1`, cubic-polynomial softmax. None of that is expressible in Triton today.

**Apr 2026 — TorchInductor adds CuTe-DSL as its fourth GEMM backend.** Inductor's autotuner picks between cuBLAS, CUTLASS C++, Triton, and CuTe-DSL. CuTe-DSL wins on NVFP4 Blackwell GEMMs.

**May 2026 — where you are.** CuTe-DSL is in public beta, scheduled to graduate by summer 2026. CUTLASS 4.5.0 is current. vLLM's hot-path GEMMs are still CUTLASS C++ but the migration is underway.

## The three layers (again, because this is the thing that gets confused)

| Layer | Lives in | What you do with it |
|---|---|---|
| **CUTLASS (library)** | `cutlass::gemm::device::Gemm<...>` and family | Pick a pre-built GEMM, parametrize element types/tile/epilogue, link it |
| **CuTe (algebra)** | `cute::Layout`, `cute::composition`, `cute::TiledMMA`, ... | Describe how a tensor is laid out in any memory; compose layouts |
| **CuTe-DSL (Python)** | `@cute.kernel`, `@cute.jit`, `cute.make_layout`, ... | Write new kernels in Python that use CuTe layouts and JIT to PTX |

A useful test: if the question is "which prebuilt kernel do I instantiate for FP8 GEMM with bias?" you are in the library layer. If the question is "how do I describe a swizzled 64×64 BF16 SMEM tile so it doesn't bank-conflict?" you are in the CuTe algebra. If the question is "how do I write that as Python that compiles to PTX?" you are in CuTe-DSL.

## Why this all matters for LLM inference

GEMM is almost everything. Estimate the FLOPs in a transformer block:

- QKV projection: 3 × `(batch*seq, hidden) @ (hidden, hidden)` — GEMM
- Attention scores: `(batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)` — batched GEMM (this is what FlashAttention fuses)
- Attention output: `(batch, heads, seq, seq) @ (batch, heads, seq, head_dim)` — batched GEMM
- O projection: GEMM
- FFN up: `(batch*seq, hidden) @ (hidden, 4*hidden)` — GEMM
- FFN down: GEMM
- Norms and activations: bandwidth-bound, small fraction of compute

For a LLaMA-shaped 70B model in BF16, **>95% of compute is GEMM**. CUTLASS covers GEMM. Triton covers GEMM too — but the last 10–15% of cuBLAS performance, and the new precisions like NVFP4 with block scaling, are where CuTe-DSL pulls ahead.

The narrower-than-it-looks observation: there are maybe four kernel families that production inference engines care about — GEMM, attention, fused norms+activations, and KV-cache management. CUTLASS handles the first two. Triton handles the third easily and the fourth adequately. CuTe-DSL handles the first two when you need the absolute peak or a new precision.

## Eight diagnostic questions

Write down your answers before continuing to submodule 02. If you can't answer one, that's the section of this README or the outer README to re-read.

1. **Why is cuBLAS not enough for LLM inference?** Hint: shapes, precisions, fused epilogues.
2. **What does "GEMM-shaped" mean and what fraction of LLM compute is GEMM-shaped?** Hint: see the breakdown above.
3. **What does a CUTLASS GEMM look like as a C++ template instantiation?** Open [vllm/csrc/cutlass_extensions/](https://github.com/vllm-project/vllm/tree/main/csrc/cutlass_extensions) and pick the FP8 GEMM. Identify the element types, the tile shape, the cluster shape, the epilogue.
4. **What is the difference between CUTLASS the library and CuTe the algebra?** Hint: a Layout is not a Gemm.
5. **Why did NVIDIA build CuTe-DSL when ThunderKittens already existed?** Hint: shared algebra with the C++ library, MLIR pipeline, NVIDIA owns CUTLASS.
6. **What did Blackwell add to the kernel-author's problem?** Hint: TMEM, tcgen05.mma issued by one thread, 2-SM cooperative MMA, NVFP4 block scaling.
7. **Why is FlashAttention-4 in CuTe-DSL and not Triton?** Hint: read the Modal writeup. Five warp specializations, TMEM-resident accumulator, cubic-polynomial softmax, `tcgen05.mma.cta_group::1`.
8. **When would you not reach for CUTLASS / CuTe-DSL — when is Triton or `torch.compile` strictly better?** Hint: bandwidth-bound ops, custom non-GEMM kernels, fast iteration on new ideas, kernels where tile shape is flexible.

Answers are in `notes.md` after you've written your own. If your answer to (8) is "never" or "always," re-read.

## Read before moving on

- [Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/) — NVIDIA's pitch for CuTe-DSL.
- [Modal: We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4) — the production proof point.
- [Ian Barber: Cute-DSL](https://ianbarber.blog/2025/07/04/cute-dsl/) — short, opinionated, useful.
