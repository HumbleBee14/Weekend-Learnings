# Level 3 — FlashAttention internals and custom attention you can actually ship

> Outer reference: [compiler-and-kernels/README.md](../README.md) · Project artifact: a custom attention variant (sliding window + sink tokens + ALiBi) implemented three ways — hand-Triton, FlexAttention, FlashInfer — and benchmarked head-to-head.

This level takes you from "I have read the FlashAttention abstract twice and it still feels like magic" to "I can read FA4's CuTe-DSL source, write a new attention variant in FlexAttention before lunch, and explain to a colleague why vLLM dispatches to FlashInfer for paged decode and to FA3/FA4 for prefill." The throughline is one kernel family — scaled dot-product attention — taken from a naive `O(N²)` memory NumPy implementation through FA2 tiling, FA3's warp-specialized Hopper variant, FA4's Blackwell rewrite in CuTe-DSL, then up the stack into FlexAttention's `score_mod`/`mask_mod` and FlashInfer's ragged-batched paged-KV serving kernels.

You finished Level 1, so you already know warps, SMs, TMA, and warp specialization in a GEMM context. You finished Level 2, so you know how `torch.compile` lowers a graph into Triton and where graph breaks cost you. Attention is where all of those skills cash out, because every modern LLM serving stack is built around one or two attention kernels, and the difference between a 700 TFLOPs/s kernel and a 1600 TFLOPs/s kernel is the difference between a profitable inference business and one that isn't.

## What you need before starting

- Level 1 done. You have written a tiled Triton matmul with autotune and at least read the warp-specialization sub-module.
- Level 2 done. You can read Inductor-emitted Triton and explain a graph break.
- You can derive the standard softmax and explain why it subtracts the max before the exp.
- You have a free Colab account (T4) for the early sub-modules. The middle of the level uses an A100 (~$2/hr on RunPod, ~$10 total). One sub-module benefits from H100; one is a B200 read-only walkthrough.

You do not need to have read the FA1/FA2 papers. We derive what we need from first principles and cite the papers for the parts we skip.

## The current attention-kernel landscape (May 2026)

Anything older than ~12 months is suspect. Here is the state of the world the week you're reading this:

- **FlashAttention-4 dropped March 5, 2026** ([arXiv 2603.05451](https://arxiv.org/abs/2603.05451)). It is written entirely in [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) embedded in Python, not CUDA C++. On B200 BF16 it reaches ~1605 TFLOPs/s (71% of peak), about 1.3× cuDNN 9.13 and 2.7× a tuned Triton attention kernel. Compile times are 20–30× faster than the FA3 C++ template path.
- **FA3 remains the standard on Hopper** ([arXiv 2407.08608](https://arxiv.org/abs/2407.08608), July 2024). Warp-specialized producer/consumer, WGMMA, FP8 with incoherent processing. 1.5–2.0× over FA2 on H100, peaking near 740 TFLOPs/s FP16 and ~1.2 PFLOPs/s FP8.
- **FlexAttention is mainline PyTorch**. `torch.nn.attention.flex_attention` takes a Python `score_mod` and/or `mask_mod` and a `BlockMask`, traces them as FX, lowers them through Inductor into Triton, and fuses them into the attention body. As of the [FlexAttention + FA4 blog (Mar 2026)](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) you can set `kernel_options={"BACKEND": "FLASH"}` and your `score_mod` is inlined into FA4's CuTeDSL softmax warps. ALiBi: 1.2–2.1× forward on GB200 vs FA2 baseline; document mask: up to 2.7×; sliding window: 1.4–2.1×.
- **FlashInfer 0.6.x is the kernel substrate under vLLM and SGLang**. It won [MLSys 2025 best paper](https://www.cs.cmu.edu/news/2025/mlsys-best-paper). Block-sparse-row KV layout, JIT-compiled attention variants, ragged batching, paged KV with arbitrary `page_size`. FlashInfer dispatches to FA2/FA3/FA4/cuDNN/TRT-LLM MHA under the hood; you call one API and get the right kernel for your hardware and shape.
- **vLLM v1 dispatch as of [Mar 2026](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)**: FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 on Ampere/Ada, FlashInfer for paged decode, and a native Triton paged-attention backend that hits 100.7% of FA3 on H100 long-decode and is the default on AMD MI300/MI325. The same Triton source runs on AMD with no changes.
- **The "Anatomy of a Triton Attention Kernel" paper** ([arXiv 2511.11581](https://arxiv.org/abs/2511.11581), Oct 2025) is the most pedagogically clean treatment of building a SOTA paged-attention kernel from scratch in Triton. We lean on it heavily in sub-modules 03 and 07.
- **2-Simplicial Attention** ([arXiv 2507.02754](https://arxiv.org/abs/2507.02754), Jul 2025) is a fresh kernel idea — trilinear attention scores over Q and two K sets — written purely in Triton, hitting ~520 TFLOPs/s on H100, rivaling FA3. We do not implement it. We do read it, because it is the cleanest example of "a researcher had a kernel idea, wrote it in Triton, and shipped it in a paper" — which is the future of this field.

A note on what is *not* in this level: training-stack ergonomics, distributed attention (Ring/Striped), sequence-parallel attention, attention with quantized KV caches in detail, MLA-style attention (DeepSeek). Those are real, important, and out of scope. The backward pass gets a survey sub-module so you know what's different and where to read more.

## How attention actually works — the minimum you need

Every transformer block does, for each head:

```
S = Q K^T / sqrt(d)            # (N, N) score matrix
P = softmax(S, axis=-1)        # (N, N) probabilities, rows sum to 1
O = P V                        # (N, d) output
```

Naive PyTorch literally materializes `S` and `P` in HBM. For sequence length `N=8192`, head dim `d=128`, dtype bf16, one head's `S` is `8192 × 8192 × 2 B = 128 MB`. With 32 heads and a batch dim, the intermediate dominates everything else. Worse: `S` is read three times (write after QK, read for max, read for sum, read for softmax * V) and `P` is read once. Four HBM round-trips of an `O(N²)` tensor. That is the whole reason FlashAttention exists.

The fix has two parts and you must understand both before any kernel makes sense:

1. **Tile.** Don't compute the whole `S`. Load a tile of Q (say 128 rows), then loop over tiles of K and V (say 128 columns each). For each `(Q_tile, K_tile, V_tile)` triple, compute a `128×128` block of scores, softmax-update an output accumulator, accumulate `P_block @ V_tile` into it. `S` and `P` never leave SRAM. Total HBM traffic drops from `O(N²)` to `O(N)`.
2. **Online softmax.** Softmax is row-wise but tiling processes columns of `S` in chunks. To compute the correct softmax you have to know the max and sum of the *whole* row before you can normalize. The online softmax trick is the math that lets you update a running `(m, ℓ, O)` state tile by tile and have the final answer be bit-exact equal to the full-row softmax. The "rescale O when the max changes" step is the heart of it. We derive it with worked numbers in sub-module 02.

Everything else in this level — FA2, FA3, FA4, FlexAttention, FlashInfer — is a refinement of these two ideas. FA2 picks the right loop order (outer over Q tiles, inner over KV tiles) and gets the indexing right for GPUs. FA3 overlaps the GEMM and the softmax across producer/consumer warps so the tensor cores never wait on the SFU exponentials. FA4 deepens the pipeline to five stages, software-emulates exp() on the FMA path (Blackwell has 2× the FMA-to-SFU ratio of Hopper), and skips the rescale-O step when the running max didn't change. FlexAttention lets you inject a Python `score_mod` that gets fused into the softmax warp. FlashInfer lets you do all of the above on a ragged batch with a paged KV cache.

If you can hold those two ideas — tile + online softmax — in your head, the rest of this level is decoration.

## What you build, sub-module by sub-module

| # | Folder | What you build | Hardware |
|---|---|---|---|
| 01 | [01-attention-from-scratch](01-attention-from-scratch/) | NumPy reference attention; measure the O(N²) memory wall | none |
| 02 | [02-online-softmax](02-online-softmax/) | Derive online softmax with worked numbers; standalone Python demo | none |
| 03 | [03-fa2-tiling](03-fa2-tiling/) | FA2 in NumPy, then a forward-only FA2 in Triton; bit-equal vs reference | T4 |
| 04 | [04-fa3-hopper-deltas](04-fa3-hopper-deltas/) | Warp-specialized FA in Triton; producer/consumer connection to Level 1 | H100 (or read trace) |
| 05 | [05-fa4-blackwell-walkthrough](05-fa4-blackwell-walkthrough/) | Read-only walkthrough of FA4 via Modal's reverse-engineering blog | B200 optional |
| 06 | [06-flexattention-custom](06-flexattention-custom/) | ALiBi + sliding window + document mask via `score_mod`/`mask_mod` | A100 |
| 07 | [07-flashinfer-ragged-paged](07-flashinfer-ragged-paged/) | Ragged batching + paged KV cache; vLLM-style block tables | A100 |
| 08 | [08-backward-pass-survey](08-backward-pass-survey/) | Read-only survey: dQ/dK/dV recomputation, varlen, determinism | A100 optional |
| -- | [_capstone-custom-attention-three-ways](_capstone-custom-attention-three-ways/) | Sliding window + sink-token + ALiBi attention in three implementations; benchmarked | A100 |

Sub-modules 01 and 02 are no-skip foundations and run on a laptop CPU. Sub-module 03 is the heart of the level — once you have a working FA2 forward in Triton, the rest is variations on a theme. Sub-module 04 needs Hopper; if you don't have H100, ship the annotated trace and walkthrough. Sub-module 05 is read-only — nobody runs FA4 directly anyway, and Modal's reverse-engineering blog plus the FA4 paper is enough to understand it. Sub-modules 06 and 07 are the high-value applied work and run cleanly on A100. Sub-module 08 is short, surveys the backward pass, and points at the right reading material. The capstone is where you ship.

## Sub-module summaries

### 01 — Attention from scratch

You write the textbook attention formula in NumPy: `S = Q @ K.T / sqrt(d); P = softmax(S); O = P @ V`. Then you measure the memory of `S` for N from 512 to 8192, plot it, and write down (in `notes.md`) exactly how many HBM round-trips a naive GPU implementation would do. By the end of this you have a reference function you will use to bit-exactly verify every kernel you write for the rest of the level.

This sub-module also introduces the masking vocabulary: causal mask, padding mask, sliding window mask, document mask. We do not implement them efficiently here — we just write each one as `score = score.masked_fill(~mask, -inf)` and confirm the output. The efficient versions come in 06.

### 02 — Online softmax, derived not asserted

This is the algebra at the center of everything. You will derive, on paper, with worked numbers, the following recursion:

Given a row `[x_1, x_2, ..., x_n]` partitioned into tiles, maintain `(m, ℓ)`:
- `m_new = max(m_old, max(x_tile))`
- `ℓ_new = ℓ_old * exp(m_old - m_new) + sum(exp(x_tile - m_new))`

The `exp(m_old - m_new)` factor is the *rescale*. It corrects the old running sum for the new (larger) max. When `m_new == m_old`, that factor is `1.0` and the rescale is a no-op — which is exactly the property FA4 exploits to skip work.

You then extend this to the output accumulator. When attention sees a new tile with a larger max, the previously-accumulated `O = sum_so_far(P V)` is rescaled by the same `exp(m_old - m_new)`. We work an example with three tiles of 4 elements each, by hand, and confirm the tiled answer equals the all-at-once answer to floating-point tolerance. There is a small Python script (`online_softmax_walk.py`) that lets you step through the computation tile by tile and print the running state.

By the end you can do this on a whiteboard. The recursion is short and once you have it, FA2 is just "apply this recursion in a Triton kernel."

The canonical reference for the math is [Zihao Ye's "From Online Softmax to FlashAttention" notes](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf), but you should produce the derivation yourself before reading them — the act of working through the indices is the lesson.

### 03 — FA2 in NumPy, then in Triton

**NumPy first.** Write the tiled algorithm with the outer loop over Q tiles, inner loop over KV tiles, maintaining `(m_i, ℓ_i, O_i)` per Q row. Output must match the NumPy reference from sub-module 01 to within `1e-5` bf16-equivalent tolerance. This is ~60 lines of NumPy and zero GPU code. Get this right and the Triton port is mechanical.

**Triton next.** Port the same algorithm to Triton, one program per Q tile. `tl.load` the Q tile once; loop over KV tiles with `tl.load` inside the loop; compute `tl.dot(q, k.T)`; scale; online-softmax-update; `tl.dot(p, v)` into accumulator. Reference: the in-tree [Triton fused-attention tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py), the paper [Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581) Section 3, and the [Modal Flash Attention walkthrough](https://modal.com/blog/flash-attention-article).

You run it on T4 at N=2048, d=64. You bit-equal-check against the NumPy reference. You profile with `triton.testing.do_bench` and compare against `F.scaled_dot_product_attention(..., backend=SDPBackend.FLASH_ATTENTION)`. On T4 you will not match FA2 — T4 has no FA2 (no SM80 features); you will be in the ballpark of `F.scaled_dot_product_attention(MATH)`. On A100 you should land within 2–3× of FA2 with a single-day effort. Reaching FA2 parity with hand-Triton is a multi-week project; that is not the bar here.

**The rescale step.** When walking the inner loop, the rescale of `O_i` by `exp(m_old - m_new)` happens *before* you add the new tile's contribution. You will get this wrong on the first try; the test will tell you. This is exactly where the FA2 paper's Algorithm 1 line 10 lives. We annotate the line of Triton code that corresponds to each line of paper-pseudocode.

This sub-module also touches `make_tensor_descriptor`-based loads as an optional follow-up. On T4 they do not change performance; on H100 they unlock TMA. We flag the lines.

### 04 — FA3 deltas: warp specialization meets attention

You already wrote a warp-specialized GEMM in Level 1. FA3 is "what if we apply that exact pattern to attention." Three changes from FA2:

1. **Producer/consumer warp specialization.** One warp group runs TMA loads of K and V tiles; another runs WGMMA for `QK^T` and `PV`; a third runs the softmax (max, exp, sum, rescale). They overlap. Tensor cores never wait on the SFU exponential. This is the ping-pong scheduler.
2. **WGMMA instead of HMMA.** Hopper's warp-group MMA operates over 128 threads with much higher per-SM throughput than the per-warp MMA Ampere used. In Triton this is hidden behind `tl.dot` plus the autotune knobs.
3. **FP8 with incoherent processing.** Random Hadamard transforms applied before quantization to spread the outlier energy across the dimensions. RMSE 2.6× better than naive per-tensor FP8.

You take your Triton FA2 from sub-module 03, enable `warp_specialize=True` on the inner KV loop, and re-benchmark. On H100 expect 1.5–2× over the FA2 version. If you don't have H100, this sub-module ships an annotated `proton` trace from a known-good run and a written walkthrough you read instead. You will not be blocked downstream.

Reference: [FA3 paper Sec 3–4](https://arxiv.org/abs/2407.08608), [Tri Dao's FA3 blog](https://tridao.me/blog/2024/flash3/), [PyTorch FA3 announcement](https://pytorch.org/blog/flashattention-3/), [Colfax's FA3 deep dive](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/).

### 05 — FA4 on Blackwell: read-only walkthrough

FA4 ([arXiv 2603.05451](https://arxiv.org/abs/2603.05451), Mar 2026) is written in CuTe-DSL and targets Blackwell SM100+. We do not implement it. We read it. The spine of the walkthrough is [Modal's "We reverse-engineered Flash Attention 4"](https://modal.com/blog/reverse-engineer-flash-attention-4) blog. What you will be able to explain when you finish:

- **Five warp specializations**, not two. FA3's ping-pong (producer/consumer) becomes Load, MMA, 8× Softmax, 4× Correction, 1–2× Epilogue. The pipeline has more stages because Blackwell's async MMA can have more outstanding ops in flight.
- **Software-emulated `exp`**. Blackwell roughly doubled FMA throughput per SM but kept SFU throughput flat. FA3 routed exponentials through the SFU. FA4 approximates `2^x` on the unit interval with a cubic polynomial in pure FMAs, matching the SFU output to bf16 precision. Applied selectively (smaller head dims benefit most).
- **Conditional softmax rescaling.** The `exp(m_old - m_new)` rescale is skipped entirely when `m_new == m_old` (or close enough not to affect numerical stability). Tri Dao reported ~10× fewer correction ops on typical prompts. This is what the FA4 paper means by "asymmetric hardware scaling" — the exp/correction path got expensive relative to the matmul path, so they trimmed it.
- **2-CTA MMA.** On Blackwell, two SMs can cooperate on one MMA, halving the per-SM SMEM pressure for K and V. Triton does not yet expose this; FA4 uses it via inline PTX (`tcgen05.mma.cta_group::1`).
- **CuTe-DSL, not C++.** The whole kernel is Python with `@cute.jit`. Compile time dropped 20–30× vs the FA3 C++ template path. This is the reason this level lives *before* Level 4 (CuTe-DSL): FA4 is the motivation, Level 4 is where you learn to write your own kernels in it.

You produce a one-page write-up in `notes.md` explaining each of these to a teammate who knows FA2 but not FA3 or FA4. Useful supplementary reading: [Lambda's FA4 announcement](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet), [Colfax's FA4 post](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/).

### 06 — FlexAttention: custom attention you can ship today

FlexAttention is `torch.compile`'s answer to "I need a slightly different attention." You give it two Python callables; it lowers them via Inductor (or via CuTe-DSL into FA4 on Blackwell with `kernel_options={"BACKEND": "FLASH"}`) into the softmax warp of a fused kernel:

```python
def alibi(score, b, h, q_idx, kv_idx):
    return score - alibi_slopes[h] * (q_idx - kv_idx).abs()

def sliding_causal(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= 1024)

block_mask = create_block_mask(sliding_causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
out = flex_attention(q, k, v, score_mod=alibi, block_mask=block_mask)
```

You build three variants:

1. **ALiBi via `score_mod`**. Verify bit-exactly against a NumPy reference. Benchmark vs SDPA.
2. **Sliding-window causal via `mask_mod` + BlockMask**. With window=1024 over N=8192, ~85% of blocks are skipped. You should see ~5–8× over full SDPA on A100.
3. **Document mask**: pack multiple variable-length documents into one fixed-length sequence; each token attends only within its document. The production case for batched fine-tuning data loaders. `mask_mod` is `doc_id[q] == doc_id[kv]`.

For each variant you `torch.compile` the wrapped function and read the emitted Triton (use `TORCH_LOGS="output_code"` or `depyf` from Level 2). What did the compiler inline into the kernel body? Where are the BlockMask metadata lookups? You write this up — three sentences each.

References: [FlexAttention launch blog](https://pytorch.org/blog/flexattention/), [FlexAttention + FA4 blog](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/), [the FlexAttention paper](https://arxiv.org/abs/2412.05496), [attention-gym](https://github.com/pytorch-labs/attention-gym) for reference implementations of dozens of variants, [Colfax's FlexAttention-in-CuTeDSL guide](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/).

The hardest thing to internalize here is that **FlexAttention is not a separate kernel** — it is a programming model that compiles into one of three or four backends (Triton, FA4/CuTeDSL, FlashAttention C++) depending on hardware and what your `score_mod` does. The kernel you write in `score_mod` runs *inside* the same async pipeline that makes FA4 fast. There is no perf cliff for "custom" attention as long as your `score_mod` is pointwise.

### 07 — FlashInfer: ragged batches, paged KV, JIT dispatch

Production serving never has a clean `(B, N, d)` tensor. You have 8 sequences with lengths `[128, 4096, 512, 2048, 64, 8192, 256, 1024]`, all sharing one paged KV pool with `page_size=16` and a block table mapping logical pages to physical. FlashInfer is the kernel library that handles this.

You build:

1. **A ragged batch prefill demo.** Concatenate variable-length sequences into one flat tensor; build the `qo_indptr` and `kv_indptr` offset arrays; call `BatchPrefillWithRaggedKVCacheWrapper`. Measure throughput vs padding to the max length and calling SDPA. On realistic length distributions you should see 2–3× from not wasting compute on pad tokens.
2. **A paged-KV decode demo.** Allocate a `[max_pages, 2, page_size, num_heads, head_dim]` paged cache. Build `kv_page_indices`, `kv_page_indptr`, `kv_last_page_len`. Call `BatchDecodeWithPagedKVCacheWrapper`. Profile with `torch.profiler` — measure JIT compile time on first call (cold) vs cached subsequent calls (hot).
3. **A "what did vLLM call" trace.** Run a small vLLM server with `VLLM_USE_V1=1`, hit it with a batch of prompts of varying lengths, capture the attention dispatch path. Identify which FlashInfer entry point gets called for prefill vs decode.

References: [FlashInfer paper](https://arxiv.org/abs/2501.01005), [FlashInfer docs](https://docs.flashinfer.ai/), [FlashInfer KV layout tutorial](https://docs.flashinfer.ai/tutorials/kv_layout.html), [yadnyesh's "Dissecting FlashInfer"](https://ydnyshhh.github.io/posts/flash_infer/), [NVIDIA's FlashInfer overview](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/).

### 08 — The backward pass: what's different

Inference kernels (FA2/3/4 forward, FlexAttention forward, FlashInfer) are most of what this level builds. Training kernels are materially different and worth a focused survey so you know what's there:

- **Two separate kernels.** Forward stores `(O, L)` where `L = m + log(ℓ)` is the log-sum-exp per query row. Backward has one kernel for `dQ` and a separate one for `dK, dV`. The split is because `dQ` is summed across query rows while `dK, dV` are summed across KV rows — and you want the parallelism axis to align with the accumulation axis, which differs.
- **Recomputation.** Backward re-computes `S` and `P` tile-by-tile from Q, K, V and the stored `L`. The same online-softmax math runs in reverse. You don't store the `N×N` attention matrix at any point; this is the whole reason FA2 backward fits in SRAM.
- **`flash_attn_func` vs `flash_attn_varlen_func`.** Fixed-shape vs ragged. Training data is almost always packed (concatenated documents with EOS separators) and uses `varlen` with a `cu_seqlens` prefix-sum array. The forward signature is similar; backward uses the same `cu_seqlens` to gradient-mask across document boundaries.
- **Determinism.** Default backward is non-deterministic because `dK, dV` use atomic accumulations across the Q-tile parallelism axis. There is a `deterministic=True` flag that serializes that axis — slower, more memory, bit-reproducible.
- **FA4 backward (Mar 2026).** Uses Blackwell's 2-CTA MMA mode and TMEM-resident reductions to keep `dK, dV` partial sums in tensor memory rather than HBM round-tripping through atomics. This is one of the bigger algorithmic wins in FA4.

You do not implement backward attention in this level. You read [Tri Dao's FA2 paper Section 3.2](https://arxiv.org/abs/2205.14135), [Aleksa Gordić's ELI5 FA backward post](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad), and [ShivamPR21's "FlashAttention Backward (Parallelism)" post](https://shivampr21.github.io/posts/flash-bwd-pll-14-4-2025-kernelized/). You write a one-page summary in `notes.md` covering the four bullets above. If you want to extend this level into a Level 3.5, [the Triton in-tree tutorial 06](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) includes a backward kernel you can read line by line.

## Capstone — Custom attention three ways

The level project. You implement **sliding-window + sink-tokens + ALiBi attention** — a real variant inspired by [StreamingLLM](https://arxiv.org/abs/2309.17453) — in three implementations:

1. **Hand-Triton.** A FA2-style forward kernel with the mask logic and ALiBi bias baked in. Forward only is fine. Reuse your sub-module 03 kernel as the starting point.
2. **FlexAttention.** `score_mod` adds the ALiBi bias; `mask_mod` enforces "attend to first K sink tokens *or* last W window tokens." Build the `BlockMask` once and reuse it.
3. **FlashInfer.** Use FlashInfer's custom-mask plumbing (BSR block-sparse with explicit mask blocks) plus a custom JIT attention variant for the ALiBi score. The most production-shaped of the three.

You benchmark all three at N ∈ {2048, 4096, 8192, 16384} on A100, with window=512, sinks=4. Same dtype (bf16), same shapes, same warmup. You produce this table in [`_capstone-custom-attention-three-ways/report.md`](_capstone-custom-attention-three-ways/):

| Impl | N=2048 | N=4096 | N=8192 | N=16384 | TFLOPs/s @ 8192 | Notes |
|---|---|---|---|---|---|---|
| F.sdpa (full attention) | | | | | | OOM expected at large N |
| Your hand-Triton | | | | | | |
| FlexAttention | | | | | | |
| FlashInfer | | | | | | |

You then write the "which won and why" section — three paragraphs, not a bulleted list. Honest assessment: which would you ship in a production engine, and why? FlexAttention almost always wins on engineering effort. FlashInfer almost always wins on a real serving workload (ragged batches, paged KV). Hand-Triton wins only if you have a kernel pattern that neither of the others can express.

If your FlexAttention number is within 10% of your hand-Triton number, you have shipped production-grade code in 30 lines instead of 300. That is the lesson.

## Definition of done

- [ ] You can derive online softmax with worked numbers, on paper, without notes.
- [ ] You have a NumPy attention reference and a NumPy FA2 reference; both produce bit-equal output.
- [ ] You wrote a Triton FA2 forward that matches the NumPy reference to bf16 tolerance.
- [ ] You either ran the warp-specialized version on H100 or you read the included trace and can explain why it wins.
- [ ] You wrote one-page summaries of FA3 deltas, FA4 deltas, and the backward pass.
- [ ] You have working ALiBi, sliding-window, and document-mask via FlexAttention with bit-equal NumPy verification.
- [ ] You have a working FlashInfer ragged-batch prefill demo and a paged-KV decode demo, with numbers.
- [ ] Capstone: sliding-window + sinks + ALiBi attention implemented three ways, benchmarked, with a "which won" writeup.

## What you can do after this level

- Read FlashAttention's CUDA C++ (FA2), Tri Dao's CuTeDSL FA4, vLLM's `vllm/attention/`, and FlashInfer's templates. Not understand every line — follow the structure, find the online-softmax block, identify the warp-spec pattern, form opinions.
- Write a new attention variant for a paper or a production engine in FlexAttention in an afternoon.
- Explain to a colleague why their training run is slow at long context, with a specific kernel-level answer.
- Make architectural decisions about which attention backend to use for a given workload — and *defend* the decision with numbers from your own benchmarks.

You are not yet at the level where you write your own FA4-equivalent kernel in CuTeDSL. That is Level 4. You are at the level where ~95% of the attention variants production engines need are within your reach — and where you understand the remaining 5% well enough to know when to call someone else.

## Common pitfalls

1. **You compared softmax outputs with `==`.** Online softmax is bit-equal to the reference only with careful summation order. Compare with `torch.allclose(..., rtol=1e-3, atol=1e-3)` in bf16, tighter in fp32.
2. **You forgot to scale by `1/sqrt(d_head)`.** Symptom: softmax is too peaked, gradients vanish, the bf16 kernel "works" but the fp32 reference disagrees. Always grep for `softmax_scale` or `1/sqrt(d)` first when debugging.
3. **You applied the rescale on `O` *after* adding the new tile.** Order matters: rescale `O` by `exp(m_old - m_new)` *first*, then add `P_new @ V_new`. Get this backwards and your output drifts on every tile boundary where the running max changes.
4. **You benchmarked FlexAttention without `torch.compile`.** Bare `flex_attention(...)` is slow — the whole point is that `torch.compile` fuses it. Always benchmark the compiled version.
5. **You created a new `BlockMask` every forward pass.** `create_block_mask` is expensive (it's a separate kernel). Build it once for a given shape and cache it.
6. **You called FlashInfer wrappers without `plan()`.** FlashInfer's batched wrappers need a `plan(...)` call with the shape metadata before `run(...)`. The plan output is cached against the metadata signature. Missing `plan()` shows up as silently wrong outputs or shape errors.
7. **You believed the FA4 numbers from the paper on a different GPU than yours.** FA4 is Blackwell-only. On Hopper, FA3 is the right number to compare against. State your GPU on every measurement.
8. **You mistook FlexAttention for a kernel.** FlexAttention is a *programming model* that compiles to a kernel. There is no `libflexattention.so`. When something is slow, look at what backend got selected (`TORCH_LOGS="output_code"`).

## Resources

**Foundational reading (do these first):**
- [Zihao Ye — From Online Softmax to FlashAttention notes](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) — the cleanest derivation of the math you need for sub-module 02.
- [Modal — What is Flash Attention?](https://modal.com/blog/flash-attention-article) — best plain-English walkthrough of FA1/FA2.
- [Tri Dao's FA3 blog](https://tridao.me/blog/2024/flash3/) — short and authoritative on the FA3 deltas.

**The current state of the art (May 2026):**
- [FlashAttention-4 paper — arXiv 2603.05451](https://arxiv.org/abs/2603.05451) (Mar 5, 2026).
- [Modal — We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4) (Sep 2025 preprint discussion + Mar 2026 update). The spine of sub-module 05.
- [PyTorch — FlexAttention + FlashAttention-4: Fast and Flexible](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) (Mar 2026).
- [Colfax — FA4 algorithm and kernel pipelining](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/).
- [Lambda — FA4 on the Blackwell platform](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet).
- [Anatomy of a Triton Attention Kernel — arXiv 2511.11581](https://arxiv.org/abs/2511.11581) (Oct 2025).
- [vLLM Triton Attention Backend Deep Dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) (Mar 2026).

**The papers (read sections, not whole things):**
- [FA2 — arXiv 2205.14135](https://arxiv.org/abs/2205.14135) Section 3.
- [FA3 — arXiv 2407.08608](https://arxiv.org/abs/2407.08608) Sections 3–4.
- [FlashInfer — arXiv 2501.01005](https://arxiv.org/abs/2501.01005), MLSys 2025 best paper.
- [FlexAttention — arXiv 2412.05496](https://arxiv.org/abs/2412.05496).
- [StreamingLLM (attention sinks) — arXiv 2309.17453](https://arxiv.org/abs/2309.17453) for the capstone motivation.
- [PagedAttention / vLLM — arXiv 2309.06180](https://arxiv.org/abs/2309.06180).
- [Fast and Simplex: 2-Simplicial Attention — arXiv 2507.02754](https://arxiv.org/abs/2507.02754) — read for the kernel idea, not for the math.

**Production kernels to read:**
- [Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention) — `csrc/flash_attn/` for FA2 in CUDA, `hopper/` for FA3, `flash_attn_interface.py` for the Python boundary.
- [FlashInfer source](https://github.com/flashinfer-ai/flashinfer) — `flashinfer/jit/` for the JIT templates, `csrc/flashinfer/attention/` for the kernels.
- [vLLM attention backends](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends) — `flash_attn.py`, `flashinfer.py`, `triton_attn.py`.
- [SGLang attention backends](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md).
- [PyTorch FlexAttention source](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py).
- [attention-gym](https://github.com/pytorch-labs/attention-gym) — reference implementations of dozens of variants.

**Tooling:**
- `F.scaled_dot_product_attention` with `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION | EFFICIENT_ATTENTION | MATH | CUDNN_ATTENTION)` for backend selection.
- `torch.profiler` with `record_shapes=True` for attention dispatch tracing.
- `triton.testing.do_bench` for kernel timing.

Older Flash Attention tutorials (pre-2024) describe APIs that are gone. The list above is the post-FA3 / post-FlexAttention set.
