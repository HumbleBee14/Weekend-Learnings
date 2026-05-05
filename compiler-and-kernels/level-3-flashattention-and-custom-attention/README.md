# Level 3 — FlashAttention Internals + Custom Attention Kernels

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: sliding-window + ALiBi attention via FlexAttention; FlashInfer ragged batching demo

## Week goal

You know FlashAttention exists and why it's fast (from `systems-for-ml` Level 2). This week you read the tiling algorithm line by line, trace the FA2 → FA3 → FA4 evolution, and write your own custom attention variants using FlexAttention. By Friday you should be able to:

- Trace the online softmax trick and explain exactly why it enables single-pass tiled attention
- Identify the specific changes FA3 made over FA2 (warp specialization + TMA + WGMMA) and connect them to the Level 1 patterns you already learned
- Understand FA4 (March 2026) — CuTe-DSL implementation, 5-stage pipeline, Blackwell-native
- Write ALiBi and sliding-window attention using FlexAttention's `score_mod`/`mask_mod` API
- Understand FlashInfer's ragged batching system and why it exists

## Where this fits

- **Comes after:** Level 1 (Triton — you understand the kernel model) and Level 2 (torch.compile — you know how PyTorch dispatches to kernels).
- **Comes before:** Level 4 (CuTe DSL — FA4 is the first major kernel written entirely in CuTe-DSL, so understanding FA4 motivates learning CuTe-DSL).

## 2026 reality check

- **FA4 dropped March 2026** (arxiv 2603.05451). It is written entirely in CuTe-DSL (Python), not CUDA C++. On Blackwell B200 BF16 it reaches 1605 TFLOPs/s — 2.7× faster than Triton and 1.3× faster than cuDNN 9.13. For Hopper (H100), FA3 remains the standard.
- **FlexAttention** is the PyTorch-native API for custom attention patterns. It generates block-sparse Triton kernels from Python `score_mod` and `mask_mod` functions. FA4 is integrated as the FlexAttention backend on Blackwell.
- **FlashInfer 0.2.x** is the kernel substrate under vLLM, SGLang, and TRT-LLM. It handles variable-length batches (ragged tensors) via CSR format, paged KV cache via page-table attention, and JIT compilation based on actual input shapes.
- **You should not implement FA3 or FA4 yourself.** These are infrastructure-level projects maintained by expert teams. The goal is to *understand* the algorithms deeply enough to debug them, extend them via FlexAttention, and make architectural decisions around them.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | online-softmax-trick | Derive the one-pass algorithm; implement naively |
| 02 | fa2-tiling-walkthrough | Trace FA2 tile-by-tile; identify the memory access pattern |
| 03 | fa3-hopper-changes | Warp spec + TMA + WGMMA in attention context |
| 04 | fa4-blackwell-and-cutedsl | 5-stage pipeline; CuTe-DSL implementation |
| 05 | flexattention-custom-masks | ALiBi, sliding window, custom score mods |
| 06 | flashinfer-ragged-batching | CSR format, page-table attention, JIT dispatch |
| 07 | attention-variants-in-prod | Which variant runs inside vLLM/SGLang/TRT-LLM |

### 01 — `online-softmax-trick`

**The problem with naive attention.** Naive attention computes `softmax(QK^T) V`. This requires materializing `QK^T` as an `N×N` matrix, writing it to HBM, reading it back for softmax, writing the softmax result, reading it back for the matmul with V. For sequence length N=8192, this is a 256MB intermediate — 4 HBM round-trips for `float16`.

**The online softmax.** Flash attention avoids this by computing softmax *incrementally* as tiles of K are loaded. The key identity:
```
softmax([x₁, x₂, ..., x_n]) can be updated incrementally:
m_new = max(m_old, x_new)
ℓ_new = ℓ_old * exp(m_old - m_new) + exp(x_new - m_new)
```
Where `m` is the running maximum and `ℓ` is the running normalization factor. After processing all tiles, `ℓ` is the true softmax denominator.

**Build steps.** Implement this in pure Python/NumPy first (no GPU). Write a reference `softmax_attention(Q, K, V)` in NumPy and a tiled version that produces the same output. Then implement it in Triton (one program per query, iterate over K/V tiles). Profile vs `F.scaled_dot_product_attention`.

### 02 — `fa2-tiling-walkthrough`

**The FA2 algorithm.** FA2 tiles Q, K, V into blocks. The outer loop is over K/V tiles; the inner loop processes query blocks against each K/V tile, maintaining the running `(m, ℓ, O)` state in registers and shared memory. At the end of the outer loop, the output is rescaled by `1/ℓ`.

**Memory access pattern.** Q is loaded once and stays in shared memory/registers for the entire K/V loop. K and V are streamed from HBM tile by tile. The output accumulator stays in registers. Total HBM traffic: `O(N)` rather than `O(N²)`.

**What to read.** FA2 paper Section 3 (from `systems-for-ml` Level 2) + Tri Dao's original implementation. The algorithm is ~50 lines; trace each line against the pseudocode in the paper. The critical insight is the `rescale` step at each tile — understanding why and when you must rescale O is the heart of it.

**Resources.**
- [FA2 paper Section 3 — arxiv 2205.14135](https://arxiv.org/abs/2205.14135)
- [Tri Dao's FA2 CUDA source](https://github.com/dao-ailab/flash-attention/blob/main/csrc/flash_attn/flash_fwd_kernel.h)
- [Modal: step-by-step FA2 walkthrough](https://modal.com/blog/flash-attention-article)

### 03 — `fa3-hopper-changes`

**The three changes FA3 makes over FA2:**

1. **Warp specialization.** FA2 used one warp group for both load and compute, alternating. FA3 splits: producer warps run TMA loads to SMEM; consumer warps run WGMMA (Hopper's tensor core instruction). They overlap — consumers run WGMMA on tile T while producers load tile T+1. This is exactly the Level 1 warp specialization pattern applied to attention.

2. **WGMMA instead of HMMA.** WGMMA (Warp Group Matrix Multiply Accumulate) is the Hopper tensor core instruction. It operates on a full warp group (128 threads) rather than a single warp (32 threads) and enables higher throughput per SM.

3. **FP8 block quantization.** FA3 adds FP8 support with incoherent processing — random Hadamard transforms applied before quantization to reduce outlier impact. This is what enables the H100 FP8 path (740 TFLOPs/s).

**What to read.** [FA3 paper — arxiv 2407.08608](https://arxiv.org/abs/2407.08608). Focus on Section 3 (algorithm) and Section 4 (Hopper-specific). Skip the appendix on proofs.

**The connection to Level 1.** The warp specialization pattern you implemented in a GEMM kernel in Level 1 is exactly what FA3 uses for attention. TMA is TMA regardless of the operation. Once you understand it in one context, you understand it everywhere.

### 04 — `fa4-blackwell-and-cutedsl`

**FA4 (March 2026, arxiv 2603.05451).** The key changes over FA3:

1. **5-stage pipeline** (FA3 had 2-stage). Blackwell's async MMA execution model allows deeper pipelining: stages for Q load, K load, QKGEMM, softmax + V load, PVGEMM. At any moment, all 5 stages are in-flight simultaneously.

2. **Software-emulated exp().** Blackwell has 2× more FMA units relative to SFU (Special Function Unit) units than Hopper. FA3 used SFU-routed exponential. FA4 approximates `exp(x)` with a short polynomial using FMAs — mathematically identical within IEEE tolerance, but 2× faster on Blackwell.

3. **Conditional softmax rescaling.** FA3 rescales the output accumulator O at every tile even when the running max hasn't changed. FA4 skips the rescale when `m_new == m_old` — this reduces rescaling ops by ~10× on typical prompts.

4. **CuTe-DSL implementation.** The entire FA4 forward kernel is Python. Not CUDA C++. CuTe-DSL's JIT compiles it to PTX at import time. This makes FA4 trivially pip-installable and easy to extend — no C++ toolchain needed.

**FlexAttention + FA4.** PyTorch's FlexAttention uses FA4 as its backend on Blackwell. Your `score_mod` and `mask_mod` Python functions get lowered by PyTorch's compiler into CuTe-DSL tensor operations that get fused into FA4's kernel body.

**What to read.** [FA4 paper — arxiv 2603.05451](https://arxiv.org/html/2603.05451v1) + [Modal's reverse-engineering of FA4](https://modal.com/blog/reverse-engineer-flash-attention-4) — Modal's post is more accessible.

### 05 — `flexattention-custom-masks`

**The FlexAttention API.** FlexAttention exposes two Python callables:
- `score_mod(score, b, h, q_idx, kv_idx)` — takes a scalar attention score and the 4D position; returns a modified score (before softmax). Use for ALiBi, relative position biases, custom temperature.
- `mask_mod(b, h, q_idx, kv_idx)` — returns True/False for whether this position should be attended to. Use for causal, sliding window, document boundaries, prefix blocks.

```python
# ALiBi: attention score decreases linearly with distance
def alibi_score_mod(score, b, h, q_idx, kv_idx):
    distance = torch.abs(q_idx - kv_idx)
    return score - alibi_slopes[h] * distance

# Sliding window: attend only to last W tokens
def sliding_window_mask(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx) <= window_size

out = flex_attention(q, k, v,
    score_mod=alibi_score_mod,
    mask_mod=sliding_window_mask)
```

**How it compiles.** PyTorch traces your `score_mod`/`mask_mod` functions as FX graphs, lowers them through Inductor into Triton tensor operations, and fuses them into the attention kernel body. Tiles that are fully masked by `mask_mod` are skipped (block sparsity) — which is why sliding-window attention is fast even on long contexts: 90% of tiles are skipped.

**Build steps.**
1. Implement ALiBi attention with `score_mod`. Verify correctness vs the HuggingFace `bloom` model's attention implementation.
2. Implement sliding-window attention (window=512) with `mask_mod`. Benchmark vs full attention on a 4096-token sequence — the speedup should be significant.
3. Implement a custom "document boundary" mask: multiple documents packed into one sequence, each attending only within its document. This is the production use case for multi-document batch processing.
4. Run `torch.compile(flex_attention)` and read the generated kernel — what did the compiler fuse into the attention body?

**Resources.**
- [FlexAttention + FA4 PyTorch blog](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [FlexAttention tutorial](https://pytorch.org/blog/flexattention/)
- [FlexAttention examples repo](https://github.com/pytorch-labs/attention-gym)

### 06 — `flashinfer-ragged-batching`

**The ragged batch problem.** In a serving context, a batch of 8 requests might have sequence lengths [128, 4096, 512, 2048, 64, 8192, 256, 1024]. Padding to 8192 wastes compute on most requests. FlashInfer handles this with CSR (Compressed Sparse Row) format: a single flat tensor of all tokens, plus an offset array marking where each sequence starts.

**The three FlashInfer attention variants:**

1. **Single-request attention** — standard FA2/FA3 for a single (Q,K,V) triple.
2. **Batched ragged attention** — CSR-format Q, K, V; each sequence has a different KV length. The kernel dispatches different tile schedules per sequence based on sequence length.
3. **Page-table attention** — for paged KV cache (as in vLLM). K, V are stored as pages in a block table; the kernel fetches pages via a lookup table rather than contiguous access.

**JIT dispatch.** FlashInfer compiles kernels at runtime based on the actual input properties: dtype, head_dim, layout (ragged vs paged), mask type (causal, sliding, prefix). The first call compiles (slow); subsequent calls use the cached kernel (fast). This is how one library covers all the variant combinations without shipping 200 separate `.so` files.

**Build steps.**
1. Install FlashInfer. Run the [ragged batching tutorial](https://docs.flashinfer.ai/tutorials/kv_layout.html).
2. Set up a ragged batch (8 sequences, variable lengths 64–8192). Run batched attention. Measure throughput vs padded batch + standard FA2.
3. Set up a paged KV cache (page size=16). Run page-table attention. This is exactly what vLLM does internally.
4. Profile with `torch.profiler` — how much time does FlashInfer spend in JIT compilation vs execution? What's the kernel cache hit rate?

### 07 — `attention-variants-in-prod`

**What each engine actually runs:**

| Engine | Prefill | Decode (paged KV) |
|---|---|---|
| vLLM V1 (Hopper) | FlashInfer batched ragged + FA3 | FlashInfer page-table + FA3 |
| vLLM V1 (Blackwell) | FlashInfer + FA4 via FlexAttention | Same |
| SGLang (Hopper) | FA3 via custom Triton backend | FlashInfer |
| TRT-LLM | cuDNN MHA / CUTLASS MHA | FlashInfer (as of v0.16) |
| llama.cpp | GGML attention (CPU) / Metal | Same |

**The dispatch stack.** vLLM doesn't call FlashAttention directly — it calls FlashInfer's dispatcher, which selects FA2/FA3/FA4/cuDNN based on hardware capability, sequence length, head dim, and quant type. You can trace this dispatch in `flashinfer/ops/attention.py`.

## Project this week

```
compiler-and-kernels/
└── attention/
    ├── online_softmax.py          # numpy + Triton implementation
    ├── flexattention_variants.py  # ALiBi, sliding-window, document-boundary
    ├── flashinfer_ragged.py       # ragged batching + paged KV demo
    └── reports/
        └── level3-attention.md   # FA2→FA4 progression notes; benchmark table
```

**Benchmark table for the report:**

| Variant | Seq len | Throughput (TFLOPs/s) | Memory (GB) |
|---|---|---|---|
| Naive attention (PyTorch) | 4096 | | |
| FA2 (dao-ailab) | 4096 | | |
| FA3 (H100) | 4096 | | |
| FlexAttention + sliding window | 4096 (W=512) | | |
| FlashInfer ragged | 4096 avg | | |

## Definition of done

- [ ] You can derive the online softmax recursion from scratch on a whiteboard.
- [ ] You can trace the FA2 algorithm tile by tile and explain the rescale step.
- [ ] You can name the three specific changes FA3 made and connect them to Level 1's warp specialization.
- [ ] You have working ALiBi and sliding-window attention via FlexAttention with benchmark numbers.
- [ ] You have a FlashInfer ragged batching demo with numbers vs padded batching.

## Resources

- **FA2 paper** — [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135). Section 3.
- **FA3 paper** — [arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608). Sections 3–4.
- **FA4 paper** — [arxiv.org/abs/2603.05451](https://arxiv.org/html/2603.05451v1).
- **Modal: reverse-engineering FA4** — [modal.com/blog/reverse-engineer-flash-attention-4](https://modal.com/blog/reverse-engineer-flash-attention-4).
- **FlexAttention blog** — [pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention/).
- **attention-gym** — [github.com/pytorch-labs/attention-gym](https://github.com/pytorch-labs/attention-gym). Reference implementations of many FlexAttention variants.
- **FlashInfer docs** — [docs.flashinfer.ai](https://docs.flashinfer.ai/).
- **Dissecting FlashInfer** — [ydnyshhh.github.io/posts/flash_infer](https://ydnyshhh.github.io/posts/flash_infer/).
- **Anatomy of a Triton Attention Kernel** — [arxiv.org/abs/2511.11581](https://arxiv.org/html/2511.11581v1).

## What you'll be able to do after this week

> Trace the FA2 → FA3 → FA4 algorithmic progression and connect each change to a specific hardware capability. Write custom attention variants (ALiBi, sliding-window, document-boundary) using FlexAttention and understand how they compile to block-sparse Triton kernels. Use FlashInfer for ragged batching and paged KV cache access. Understand the attention dispatch chain inside vLLM and SGLang.
