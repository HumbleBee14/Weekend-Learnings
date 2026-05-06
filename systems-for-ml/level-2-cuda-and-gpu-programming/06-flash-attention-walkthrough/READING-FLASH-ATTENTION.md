# Reading FlashAttention — the production source

You implemented a minimal FA2 in Triton (`flash_attention_minimal.py`). Now read the production sources and compare. This is where the algorithm meets the real engineering.

## What we're reading

Three sources, three regimes:

1. **dao-ailab/flash-attention** — Tri Dao's reference implementation. CUDA C++. FA2 + FA3 (FA4 is in a separate Python repo).
2. **flashinfer-ai/flashinfer** — the production attention library that vLLM, SGLang, TRT-LLM all use. Mix of CUDA and Triton.
3. **vLLM's Triton FA backend** — a Triton-native FlashAttention currently shipped in vLLM, intended to eventually replace the legacy CUDA path.

## Setup

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
git clone https://github.com/flashinfer-ai/flashinfer.git
# vllm — already cloned in 05's READING-PRODUCTION-KERNELS.md
```

No build. We're reading.

## Exercise 1 — FA2 in CUDA C++

The classic. ~10,000 lines of templated CUDA C++. You're not reading it all; you're navigating it.

**File**: [`flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)

**What to find:**

### 1. The outer Q-tile loop

Search for `for` loops at the top level of the kernel. The outermost is over Q tiles — same as your `flash_attention_minimal.py`. Each iteration of this loop handles BLOCK_M rows of output.

### 2. The inner K, V tile loop

Inside the outer loop, the inner loop streams K and V tiles. Find the line that does the matmul:
```cpp
flash::gemm</*A_in_regs=*/false>(acc_s, tSrQ, tSrK, ...);
```
That's `S = Q @ K^T` for one tile.

### 3. The online softmax update

Search for `softmax_lse_O` (log-sum-exp) and `softmax_rescale_o`. These functions implement the running `(m, ℓ)` recursion — same math as your minimal version, but factored into helper functions.

### 4. The use of CUTLASS

Search for `cutlass::`. FA2 uses CUTLASS templates for the matmul (`cutlass::gemm::collective::CollectiveMma`), epilogue (`cutlass::epilogue::collective::CollectiveEpilogue`), and tensor core operations. **The matmul isn't hand-written CUDA — it's CUTLASS.** This is what production-grade kernels look like in 2026: glue + CUTLASS, not from-scratch CUDA.

### 5. The template parameter explosion

Look at the kernel signature:
```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap,
         bool Return_softmax, typename Params>
__global__ void flash_fwd_kernel(...)
```

Each `bool` template parameter creates a specialized kernel variant at compile time. There are ~512 variants compiled into the final library. This is how C++ templates handle "many flavors of attention" — at compile time, with separate specialization per combination.

**Compare to Triton**: the same flexibility comes from `tl.constexpr` parameters and runtime config selection — much less compile-time machinery.

## Exercise 2 — FlashInfer's dispatch logic

FlashInfer is the production layer above FA. It dispatches between FA2/FA3/cuDNN/CUTLASS based on workload.

**File**: [`flashinfer/python/flashinfer/prefill.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/python/flashinfer/prefill.py)

**What to find:**

### 1. The kernel cache

Search for `_cached`. FlashInfer JIT-compiles attention kernels per (dtype, head_dim, mask_type, layout) combination at runtime, then caches them. First call: slow. Subsequent: fast.

### 2. The layout selection

Search for `kv_layout`. Three values: `NHD` (num_heads, head_dim packed), `HND` (heads, num, dim), and ragged formats. The kernel chosen depends on this — same algorithm, different memory access pattern.

### 3. The page table integration

Search for `paged_kv_cache`. This is the FlashInfer equivalent of vLLM's PagedAttention block table — a level of indirection between logical and physical KV blocks.

### 4. The C++ entry point

Search for `run_paged_kv_attention`. This calls into a CUDA kernel (in `flashinfer/csrc/`). The Python file is *dispatch* logic; the kernel itself is C++.

**What to take away.** FlashInfer is the example of how production teams structure this:
- **Python**: dispatch, layout selection, kernel selection, JIT cache
- **C++ / Triton**: the actual kernel
- **CUTLASS**: the matmul primitives inside the C++ kernel

You don't pick "C++ vs Python"; you use both, with each layer doing what it's best at.

## Exercise 3 — vLLM's Triton FA

Already covered in `05/READING-PRODUCTION-KERNELS.md`. If you didn't do that exercise, do it now and pay attention to the FlashAttention algorithm specifically.

**Compare** to:
1. Your minimal Triton FA from this topic
2. dao-ailab's CUDA C++ FA2

You'll see: same algorithm, three flavors. Yours is the simplest. dao-ailab's is the fastest hand-tuned. vLLM's Triton version is the production-Python middle ground.

## Exercise 4 — FA4 in CuTe-DSL (Python!)

The newest version. Tri Dao's blog: https://tridao.me/blog/2026/flash4/. Code: https://github.com/Dao-AILab/flash-attention (FA4 branch).

**Notice the language.** FA4 is written in CuTe-DSL — Python. The author of FlashAttention chose Python over C++ for the new frontier kernel.

**Why?** Because CuTe-DSL gives you:
- Direct control over WGMMA/tcgen05 instructions (which Triton abstracts away)
- Tile-layout algebra in Python (`(M, N):(strideM, strideN)` compositions)
- Faster iteration than C++ during development

You won't read FA4 closely in this exercise — it's frontier work. But knowing it exists and is in Python is the point.

## What this whole exercise teaches

You're not learning to write FlashAttention. You're learning:

1. **The same algorithm shows up at multiple polish levels** — your 80-line Triton version, vLLM's production Triton version, dao-ailab's hand-tuned CUDA C++ + CUTLASS, and Tri Dao's CuTe-DSL FA4. They're all the same recursion. Different languages and tuning effort buy different performance, but the math is one thing.

2. **Production stacks layer Python and C++** — FlashInfer's Python dispatcher calling into CUDA C++ kernels (which use CUTLASS) is the canonical pattern.

3. **The frontier moved to Python** — FA4 in CuTe-DSL is the strongest signal that high-performance kernel work is no longer C++-exclusive.

## Honest takeaway for your career

If you're hired to "work on inference," the things you'll actually do:

- **Read** production kernels (CUDA C++ and Triton, depending on the codebase). 70% of the work.
- **Write** Triton or CUDA C++ glue around existing primitives. 20%.
- **Hand-tune** a single kernel from scratch in CUDA C++. 5–10%, and only at frontier labs.

The skill the curriculum is preparing you for is the first one — being able to navigate, audit, and modify a real serving stack. Topic 5's "Reading Production Kernels" and this Topic 6 reading exercise are the artifacts of that skill.

## References

- dao-ailab/flash-attention — https://github.com/Dao-AILab/flash-attention
- FlashInfer — https://github.com/flashinfer-ai/flashinfer
- Tri Dao FA4 blog (CuTe-DSL!) — https://tridao.me/blog/2026/flash4/
- Modal — reverse engineering FA4 — https://modal.com/blog/reverse-engineer-flash-attention-4
- vLLM Triton attention deep dive — https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html
- The Anatomy of a Triton Attention Kernel — https://arxiv.org/abs/2511.11581
