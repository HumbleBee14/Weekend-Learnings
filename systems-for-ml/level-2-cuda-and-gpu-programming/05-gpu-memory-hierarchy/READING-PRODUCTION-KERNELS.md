# Reading production kernels

The skill that matters in industry isn't writing yet another softmax kernel. It's **reading the kernels in the codebase you'll be working in**. This note walks you through three real production CUDA C++ kernels and what each one teaches about the memory hierarchy.

You won't run anything here. You'll clone two repos and read code.

## Setup

```bash
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/linkedin/Liger-Kernel.git
```

That's it. No build needed. We're reading.

## Exercise 1 — vLLM PagedAttention (CUDA C++)

The kernel that put vLLM on the map. Implements paged KV cache attention — the algorithmic innovation behind vLLM's continuous batching.

**File**: [`vllm/csrc/attention/attention_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu)

**What to look for as you read:**

### 1. The block table indirection

Find the line that looks like:

```cpp
const int physical_block_number = block_table[seq_idx * max_num_blocks_per_seq + block_idx];
```

This is paging in action. Logical block numbers (per-sequence) get translated to physical block numbers (chip-wide) via a lookup table — same idea as virtual memory in an operating system, applied to KV cache.

The memory hierarchy implication: **KV cache lives in HBM** (it's too big for SMEM), but the *block table* is small enough to live in L2 cache after the first read. So lookups are fast even though the KV blocks themselves are HBM reads.

### 2. The shared memory buffers

Search for `__shared__`. You'll find buffers like:
```cpp
__shared__ float red_smem[2 * NUM_WARPS];
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

These are exactly the SMEM tiles from Topic 5's CONCEPTS.md, applied to attention. Q vectors are loaded once per query group and reused across many keys.

### 3. The reduction pattern

Search for `__shfl_xor_sync` and `__shfl_down_sync`. These are warp shuffles — the same pattern from `02-first-cuda-kernels/softmax.cu`. They reduce within a warp without going through shared memory.

You'll see the exact tree-reduction shape from Topic 2:
```cpp
for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
}
```

### 4. Coalesced loads via vectorized types

Search for `Quant_vec`, `K_vec`, `V_vec`. These are typedef'd vector types like `float4` or `bfloat16x8` — the vectorized loads from `02-first-cuda-kernels/relu_vec4.cu`, applied to KV cache reads.

**What to take away.** This kernel is ~1200 lines. Most of it is:
- Block table indirection (paging logic)
- SMEM tiling for Q
- Warp shuffles for max/sum reductions
- Vectorized loads for K and V

Every concept from Topics 1–5 of this level shows up. You can read this kernel because you understand the model. **That's the bar.** You don't need to be able to write this from scratch; you need to be able to find a bug in it.

### Why is this in CUDA C++ and not Triton?

Two reasons:
1. **Historical** — written in 2023 before Triton was production-ready for paged attention.
2. **Performance edge cases** — at the time, hand-tuned CUDA C++ beat Triton on this specific workload.

The vLLM team is migrating. The newer "Triton attention backend" ([Mar 2026 blog](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)) is the eventual replacement. Reading both lets you compare them directly.

## Exercise 2 — Liger-Kernel RMSNorm (Triton)

The same kind of kernel, in Triton, written in 2024. Compare to PagedAttention's CUDA C++ to feel the difference.

**File**: [`Liger-Kernel/src/liger_kernel/ops/rms_norm.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py)

**What to look for:**

### 1. Where the SMEM tiling is

Trick question — there's no explicit SMEM allocation. Triton handles it. But the `BLOCK_SIZE: tl.constexpr` parameter implicitly determines tile size, and the compiler decides whether to put intermediate values in SMEM or registers.

You can verify this by running with `TRITON_DEBUG=1` and looking at the generated PTX — you'll see `ld.shared` and `st.shared` instructions emerge.

### 2. The reduction

Look for `tl.sum(...)` and `tl.max(...)`. These are tile-level reductions; the compiler emits warp shuffles + SMEM exactly like the PagedAttention kernel did, but you don't see it.

### 3. The line count

The whole file is ~150 lines, including forward + backward + Python wrapper. PagedAttention's CUDA C++ is ~1200 lines for forward only. The lessons are the same; the *language* is doing more work for you in Triton.

**What to take away.** Same memory hierarchy story (registers → SMEM → HBM, fused reads/writes), much shorter source. This is why new kernel work moved to Triton.

## Exercise 3 — vLLM's Triton attention backend (the future)

The 2026 replacement for PagedAttention. Already at production parity for many shapes, ahead in some.

**File**: [`vllm/vllm/attention/ops/triton_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_attention.py) and surrounding directory.

**What to look for:**

### 1. The autotune configs

Search for `@triton.autotune`. Compare the config list to what you'd hand-pick. The configs sweep `(BLOCK_M, BLOCK_N, num_warps, num_stages)` — exactly what you'd manually tune in CUDA C++, but here it's expressed as data and the autotuner picks per-shape.

### 2. The same FlashAttention algorithm

You'll recognize the structure: outer loop over Q tiles, inner loop over K, V tiles, running `(m, ℓ)` state. **It's the same algorithm** as `06-flash-attention-walkthrough/flash_attention_minimal.py` — just at production polish.

### 3. The block-table integration

Look for how `block_table` is passed in and used. This is the Triton equivalent of the paging logic from Exercise 1. Same idea, different syntax.

**What to take away.** Triton can do everything PagedAttention CUDA C++ does, and it's becoming the default at vLLM. The Triton path is shorter, easier to audit, and gets new optimizations (like Triton 3.2's automatic warp specialization) for free.

## What this exercise teaches

Reading production code is the skill that doesn't fit in topic-sized chunks. It's the difference between:

- **"I learned CUDA"** — wrote 5 toy kernels in a tutorial
- **"I can work on vLLM"** — can navigate `csrc/`, find the relevant kernel for a bug report, understand its data flow, propose a fix

Most jobs that touch GPU kernels are the second one. Few involve writing a new kernel from scratch.

After working through these three files you'll have:
- Read 1500+ lines of production CUDA C++ and Triton
- Recognized every concept from Topics 1–5 in real production code
- Felt the difference in line count between hand-tuned CUDA C++ and Triton
- A baseline for what "professional kernel code" looks like

## Suggested workflow

1. **Read PagedAttention** with the headers section first, then `paged_attention_kernel<>` itself. Skim — don't try to fully understand on first pass. Note where each Topic concept appears.
2. **Read Liger-Kernel RMSNorm** in full. Annotate which lines correspond to which Triton primitive.
3. **Diff in your head**: same kind of computation, very different code. What changed?
4. **Read vLLM's Triton attention** to see the bridge — production-grade Triton in a real serving stack.

Time budget: 2-3 hours. No code to write.

## References

- vLLM `csrc/` — https://github.com/vllm-project/vllm/tree/main/csrc
- vLLM Triton attention backend deep dive — https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html
- Liger-Kernel — https://github.com/linkedin/Liger-Kernel
- The Anatomy of a Triton Attention Kernel — https://arxiv.org/abs/2511.11581 (paper that builds a Triton-only paged attention to 105% of SOTA)
