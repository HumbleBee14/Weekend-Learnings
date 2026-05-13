# Level 1 — Triton, from zero to writing kernels people ship

> Outer reference: [`compiler-and-kernels/README.md`](../README.md)

This level takes you from "I have never written a GPU kernel" to "I can write the kernels that vLLM, SGLang, Liger-Kernel, and Unsloth ship in their hot paths." That sounds large for one level — it works because Triton was designed exactly to compress this gap. The price you pay is that you have to understand a few things deeply: how a GPU executes, what memory bandwidth means in numbers, and what the compiler does for you vs. what you have to tell it.

The throughline is one operation — **RMSNorm** — taken from a naive Triton kernel pulling 11% of peak HBM bandwidth all the way to a fused, autotuned, warp-specialized version pulling 88%+. Same op, same numbers, same profiler. Each new technique earns its place by closing part of that gap, and you see the number move.

By the end of this level the level capstone is a **fused RMSNorm+RoPE kernel benchmarked head-to-head against Liger-Kernel's production version**. If you match Liger's bandwidth utilization, you have written code that would be accepted into their repo — that is the actual bar.

## What you need before starting

- You have written Python.
- You have a vague idea what a GPU is — "it's parallel and has its own memory" is enough.
- You have either a Google Colab account (free T4 GPU) or any cloud GPU. **Everything except the optional Blackwell section runs on a free T4.** A few sections benefit from an H100 or B200 if you have one, and are clearly marked.
- You finished `systems-for-ml` levels 1–3, or you have equivalent: you can read an `nsys` / `ncu` trace and know what HBM bandwidth means.

You do **not** need CUDA C++ experience. Triton is reachable without it. Some CUDA vocabulary leaks in here (warp, SM, shared memory, tensor core) — we build each term the first time we use it.

## The current Triton landscape (May 2026)

A learner who reads any tutorial older than ~12 months will absorb wrong APIs. The state of the world right now:

- **Triton 3.7.0** is the current stable release (May 7, 2026). It ships with recent PyTorch nightlies. The release cadence is roughly every 2–3 months.
- **Warp specialization is mainline.** You enable it with `tl.range(..., warp_specialize=True)` and the autotune flags `num_consumer_groups` / `num_buffers_warp_spec`. The Tawa paper (arXiv 2510.14719) is the formal description; the implementation was upstreamed in [PR #6288](https://github.com/triton-lang/triton/pull/6288) and follow-ups. A lot of older tutorials describe a manual warp-id split — that route exists but is not what people use.
- **TMA (Tensor Memory Accelerator)** is exposed as `tl.make_tensor_descriptor(...)` followed by `desc.load([offs])` / `desc.store(...)`. Supports 2–5D tensors. Lowers to TMA on Hopper/Blackwell; falls back cleanly on older hardware.
- **Blackwell (B200, SM100)** added Tensor Memory (TMEM) and the tcgen05 MMA family. Triton uses these under the hood — you mostly get speed for free, but a few advanced patterns (2-SM cooperative MMA, TMEM-resident pipelined epilogues) are not yet fully exposed in Triton. This is why FlashAttention-4 is written in CuTe-DSL, not Triton.
- **AMD parity is real now.** Warp specialization on AMD (called "wave specialization") and TDM (AMD's TMA analog) landed in Triton 3.7. vLLM's Triton paged-attention backend is the **default on AMD MI300/MI325**, with the same source running on both vendors.
- **Gluon** is a lower-level dialect inside Triton for hardware-specific control. You should know it exists; you do not need it for this level.

We pin **Triton 3.7.0** for this level. If you have to use an older version, the API differences are flagged where they matter.

## How a GPU actually executes — the minimum you need

Most kernel pain comes from a vague mental model of the hardware. Five things, plain English:

**A GPU is a stack of Streaming Multiprocessors (SMs).** An H100 has 132 SMs; a B200 has 148; a T4 has 40; an MI300X has 304 "CUs" which are the AMD equivalent. Each SM is an independent worker. To use the whole GPU, your work has to split into pieces that each fit on one SM and you have to launch enough pieces.

**An SM runs work in warps of 32 lanes.** A warp is the smallest unit the hardware schedules. All 32 lanes in a warp execute the same instruction at the same time — they share an instruction pointer. If your code says `if x > 0: A else: B`, lanes where the condition is true execute A while the others wait, then vice versa. That stall is called *warp divergence*. (AMD calls them "wavefronts" of 64 lanes — same idea, different number.)

**Memory hierarchy, roughly in order of speed and size:**

| Level | Size per SM | Latency | Bandwidth |
|---|---|---|---|
| Registers | ~256 KB | 1 cycle | unlimited |
| Shared memory (SRAM) | 100–228 KB | ~30 cycles | ~10 TB/s |
| L2 cache | 50 MB (shared, H100) | ~200 cycles | ~5 TB/s |
| HBM (global memory) | 80 GB | ~500 cycles | 3.4 TB/s |

The point of every fast kernel is: **load from HBM once, do as much work as possible in registers and SRAM, write back to HBM once.** When you hear "this kernel is memory-bound" or "we fused these ops to save HBM round-trips," this table is what is meant.

**Tensor cores are specialized matrix-multiply units inside each SM.** On H100 they do FP16 matmul at ~989 TFLOPS — a couple of orders of magnitude faster than the regular FP32 ALUs. On Blackwell, tensor cores got new precisions (NVFP4, MXFP8) and new memory (TMEM) sitting right next to them. To use them you call `tl.dot` in Triton; the compiler picks the right tensor-core instruction.

**Async copies (TMA on NVIDIA, TDM on AMD).** Modern GPUs have hardware that can copy a tile from HBM to shared memory in the background while your compute keeps running. Before this hardware, every load had to be staged by the warp itself — wasting compute cycles waiting on memory. TMA changed the game: one instruction kicks off a multi-KB tile copy, and your warp goes back to computing. This is the single most important hardware feature for modern kernels.

Everything else in this level is a consequence of these five facts.

## What Triton is and what it gives you

CUDA C++ asks you to write one thread's worth of work, then think about how 1024 of them coordinate. Triton asks you to write one *tile* — a block of data — and the compiler handles the threads, the shared-memory allocation, the vectorization, and increasingly the tensor-core scheduling. You write code that looks like NumPy on small blocks, and you get a real GPU kernel.

The trade is: you give up some control. You cannot ask for "this exact warp to do this exact thing" without dropping to PTX or Gluon. You get back: 5–10× shorter source code, kernels that retarget to AMD/Intel/RISC-V from the same `.py` file, and a compiler that has spent the last two years getting smarter at warp specialization and async pipelining.

Concrete: the vLLM paged-attention kernel is **~800 lines of Triton** and hits parity with FA3 (which is ~70k lines of CUDA C++) on H100 long-context decode. That ratio is the reason Triton matters.

## What you build, topic by topic

| # | Folder | What you build | Hardware |
|---|---|---|---|
| 01 | `01-gpu-mental-model` | Diagrams + a one-page write-up that tests your understanding before any code | none |
| 02 | `02-first-triton-kernel` | Vector add, elementwise scale, then a softmax — to learn the language | free T4 |
| 03 | `03-rmsnorm-bandwidth-journey` | RMSNorm from 11% to 88% peak HBM bandwidth, in 5 steps | free T4 |
| 04 | `04-tiled-matmul-and-autotune` | Tiled GEMM with `make_tensor_descriptor`, then proper autotune with pruning | free T4 (H100 optional) |
| 05 | `05-tma-and-warp-specialization` | Warp-specialized GEMM and attention; producer/consumer with `tl.range(warp_specialize=True)` | H100 or B200 |
| 06 | `06-persistent-kernels` | Persistent matmul with dynamic tile assignment; CUDA-graph compatible | T4 works, H100 better |
| 07 | `_capstone-fused-rmsnorm-rope` | Fused RMSNorm+RoPE vs Liger-Kernel head-to-head | free T4 |

Sub-modules 01 and 02 are no-skip foundations. Sub-modules 03 and 04 are the heart of the level — the RMSNorm bandwidth journey teaches you to make a kernel fast, and tiled-matmul teaches you the GEMM-shaped vocabulary you need everywhere else. Sub-module 05 needs Hopper or Blackwell; if you do not have one, read the included annotated trace and skip the run — you will not be blocked. Sub-module 06 is short and unlocks the CUDA-graph story that L2 of this track depends on.

### 01 — GPU mental model

Read the GPU section above and the linked deep-dive in [`01-gpu-mental-model/CONCEPTS.md`](01-gpu-mental-model/CONCEPTS.md). Then answer the eight diagnostic questions in that folder's README. If you cannot answer them, code in later sub-modules will feel like spell-casting. If you can, you are ready.

### 02 — Your first Triton kernel

Three kernels in increasing depth: vector add → fused elementwise scale-and-add → softmax over a row. By the end you have written `@triton.jit`, used `tl.program_id`, `tl.arange`, `tl.load` with masks, `tl.store`, `tl.sum`, `tl.exp` — the vocabulary that covers maybe 80% of what you'll ever write. Each kernel is short (~30 lines) with an annotated explanation of every line.

The softmax kernel is where you meet your first real subtlety: **online softmax** (computing max and sum in a single pass for numerical stability). This idea recurs every time we touch attention.

### 03 — The RMSNorm bandwidth journey

This is the central exercise of the level. You write the same operator five times, and you watch one number — fraction of HBM peak bandwidth achieved — climb from 11% to 88%.

The five versions:

1. **Naive row-per-program.** One Triton program per row of the input. ~11–15% of peak. The bottleneck is small tile size — each SM does too little work per launch.
2. **Vectorized loads.** `tl.load` of 8 elements at a time (`tl.float32 * 8` cast). ~30%. We are now reading memory in efficient transactions but still doing one pass for the mean and another for the normalization.
3. **Single-pass with online stats.** Compute the mean and the normalization in one pass over the input. ~55%. HBM traffic is halved.
4. **Autotuned BLOCK_SIZE.** Let `@triton.autotune` pick the tile size with a pruning function so we don't sweep nonsense configs. ~75%.
5. **Persistent + cached writeback layout.** Reuse the same SM for multiple rows, write outputs in a cache-friendly order. ~88%.

Each step is its own file. Each step has a profiler trace (DRAM throughput, SM occupancy, L2 hit rate) saved as a markdown table in the folder. You run them, get your own numbers, and explain in your `notes.md` why each step moved the needle.

Subhadip Mitra's writeup "From 11% to 88% peak bandwidth" is the inspiration here — but we go a step further by autotuning correctly and adding the persistent pattern. By the end you have the canonical bandwidth-bound kernel template you can apply to every elementwise+reduction op (LayerNorm, RMSNorm, GeGLU, SwiGLU, the residual streams).

### 04 — Tiled matmul, autotune that doesn't waste your money

GEMM is the shape every learner must understand because attention, MLPs, projections, and embeddings all reduce to GEMM variants. You write three matmul kernels:

1. **Tiled with explicit `tl.load`** of (M, K) and (K, N) blocks into registers, `tl.dot` into accumulator. This is the form most tutorials stop at. You measure: roughly 30% of cuBLAS on T4 / H100.
2. **Tiled with `tl.make_tensor_descriptor`** for both inputs. The descriptor handles strides, masking, and on Hopper+ lowers to TMA. Same code, ~70% of cuBLAS on H100.
3. **Autotuned with `early_config_prune`.** Naive autotune sweeps ~200 configs and many are nonsense (BLOCK_M=256 × num_warps=1 won't fit in registers). You write the pruning function that filters configs by register pressure, SMEM budget, and tensor-core alignment. Time to tune drops from ~hours to ~minutes; the winner is the same or better.

You read the `triton.proton` profile of the winning config and write three sentences explaining why it won. Not which config — *why*. (Hint: it almost always comes down to keeping the tensor cores fed without spilling registers.)

This sub-module is where you also see `torch.compile`'s Inductor-emitted Triton for the same shape. You compare your hand-written kernel to what Inductor produced. Sometimes you win; usually Inductor is within 5%. Either outcome is informative and you write up which it was and why.

### 05 — TMA and warp specialization

The single highest-leverage hardware feature on Hopper and Blackwell.

You start by re-reading your tiled matmul from sub-module 04 with TMA descriptors. Then you turn on warp specialization with one line:

```python
for k in tl.range(0, K, BLOCK_K, warp_specialize=True, num_stages=4):
    a = a_desc.load([offs_m, k])
    b = b_desc.load([k, offs_n])
    acc = tl.dot(a, b, acc)
```

The compiler partitions the loop body across producer and consumer warp groups: producer warps issue TMA loads, consumer warps run MMAs. The two run in parallel — consumers compute on tile *n* while producers fetch tile *n+1*. This is the pattern that gave FlashAttention-3 its 1.5–2× speedup over FA2 and is now what every fast GEMM and attention kernel uses.

You measure: tensor-core utilization (peak FP16 TFLOPS), HBM-to-SRAM async occupancy, and the speedup over the non-warp-specialized version. On H100 expect ~1.3–1.5×. On B200 expect more, plus a deeper discussion of TMEM and tcgen05 — these are the new pieces of Blackwell, and we walk through what Triton exposes vs. what still requires CuTe-DSL.

If you don't have a Hopper or Blackwell GPU, the sub-module ships an annotated `proton` trace from an H100 run and a written walkthrough. You read it instead of running it. You will not be blocked in later levels.

### 06 — Persistent kernels and CUDA-graph compatibility

A non-persistent kernel launches `grid_size = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)` programs and the hardware scheduler doles them out. Two problems:

- **Variable grid size breaks CUDA graphs.** Every time the shape changes, you need a new graph. For LLM inference with variable sequence lengths, this means re-capturing constantly.
- **Launch overhead is real.** ~5–10 µs per launch. For decode (one token at a time) this is a meaningful fraction of total latency.

A persistent kernel launches exactly `num_SMs` programs (so the grid is fixed for the device) and each program loops over multiple tiles internally, picking the next tile via an atomic counter or precomputed schedule. The hardware never has to schedule again — your kernel does the scheduling itself.

You write a persistent version of the tiled matmul from sub-module 04, then verify it captures into a CUDA graph cleanly. You measure decode-shape (M=1, M=8) latency vs. non-persistent — expect a 2–5× win on these small shapes.

This pattern is what vLLM v1, the Triton paged-attention kernel, and the PyTorch grouped-GEMM MoE all use. The technique is small; the implications for production inference engines are large. Level 2 of this track (`torch.compile` internals) builds the "piecewise CUDA graph" pattern directly on top of this.

## Capstone — Fused RMSNorm+RoPE vs Liger-Kernel

Every transformer block in a LLaMA-shaped model passes every token through RMSNorm and RoPE multiple times. Unfused, each is a separate kernel with its own HBM round-trip. Fused, you read the input once, do all the math in registers, write the output once. For decode-heavy workloads this is one of the highest-impact fusions you can do.

The capstone in [`_capstone-fused-rmsnorm-rope/`](_capstone-fused-rmsnorm-rope/) is structured as:

1. Read [Liger-Kernel's `ops/rms_norm.py` and `ops/rope.py`](https://github.com/linkedin/Liger-Kernel/tree/main/src/liger_kernel/ops). Note every non-obvious decision they made — precision choices, where they use `tl.where` vs branching, how they handle the backward pass.
2. Write your own fused kernel from scratch. Forward only is the minimum bar; backward is the stretch goal.
3. Benchmark against four references: eager PyTorch (unfused), `torch.compile`, Liger-Kernel, and Liger-Kernel's `KernelBench` reference if you can. Same shape, same dtype, `triton.testing.do_bench` with warmup.
4. Profile with `triton.proton`. Confirm one HBM read of the input, one HBM write of the output. If you see more, find out why.
5. Write a one-page report: where you matched Liger, where you fell short, what you'd change next. If you exceeded Liger, double-check measurement (warmup, dtype, shape) before claiming the win.

The benchmark table you produce:

| Kernel | dtype | Hidden dim | Tokens | GB/s | % HBM peak | vs Liger |
|---|---|---|---|---|---|---|
| Eager PyTorch (unfused) | bf16 | 4096 | 32×2048 | | | baseline |
| `torch.compile` | bf16 | 4096 | 32×2048 | | | |
| Liger-Kernel | bf16 | 4096 | 32×2048 | | | reference |
| Yours, no autotune | bf16 | 4096 | 32×2048 | | | |
| Yours, autotuned | bf16 | 4096 | 32×2048 | | | |
| Yours, persistent + autotuned | bf16 | 4096 | 32×2048 | | | |

If your final number is within ±5% of Liger on percent of HBM peak, you have produced production-grade code. That is the bar.

## Definition of done

- [ ] You can explain, without notes: warps, SMs, the memory hierarchy, why HBM bandwidth dominates elementwise+reduction kernels.
- [ ] You wrote vector add, scale-add, softmax, RMSNorm (5 versions), and tiled matmul in Triton.
- [ ] Your RMSNorm bandwidth journey log shows the 11% → 88% progression with profiler evidence for each step.
- [ ] Your tiled matmul reaches >70% of cuBLAS on the hardware you have (H100 if available, T4 otherwise — scaled to T4's tensor-core limits).
- [ ] You either ran the warp-specialization sub-module on H100/B200 or you read the included trace and can explain why warp specialization wins.
- [ ] You have a working persistent matmul that captures into a CUDA graph.
- [ ] Capstone: fused RMSNorm+RoPE kernel within ±5% of Liger-Kernel on % HBM peak, with a report explaining your numbers.

## What you can do after this level

You can pick up vLLM's `vllm/attention/ops/triton_paged_attention.py`, SGLang's fused MoE kernel, or Liger-Kernel's fused linear cross-entropy and read them — not understand every line on the first pass, but follow the structure, identify the warp-specialization pattern, find the TMA descriptors, and form an opinion about what each design choice is buying. You can also write your own kernel for a new fused op a colleague needs and have a reasonable expectation of getting within 10% of a hand-tuned reference on the first try.

You are not yet at the level of someone who writes CuTe-DSL kernels for Blackwell with TMEM-resident pipelined epilogues — that is Level 4. You are at the level where 90% of the kernels production inference engines need are within your reach.

## Resources

The current and useful set, all 2024–2026 unless marked. Older Triton material exists in abundance and will mislead on the post-3.4 APIs.

**Foundational reading.** Read in this order:
- [Triton language overview](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) — the official starting point.
- [`triton-lang/triton/python/tutorials/`](https://github.com/triton-lang/triton/tree/main/python/tutorials) — the in-tree tutorials. Tutorial 03 (matmul) and 06 (fused attention) are the most useful.
- [GPU MODE Lecture 14 — A Practitioner's Guide to Triton](https://github.com/gpu-mode/lectures/tree/main/lecture_014) — single best 90-minute primer.

**The current state of the art.** These are what we are training you toward:
- [Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581) (Oct 2025) — the definitive walkthrough of writing a SOTA paged-attention kernel in Triton from scratch.
- [vLLM Triton Attention Backend Deep Dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) (Mar 2026) — how vLLM's ~800-LoC kernel reaches FA3 parity.
- [PyTorch: Warp Specialization in Triton — Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) (Jan 2026).
- [Tawa paper — arXiv 2510.14719](https://arxiv.org/abs/2510.14719) — the formal description of how warp specialization is implemented.
- [PyTorch: Persistent Cache-Aware Grouped GEMM for MoE](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) — the persistent-kernel pattern in production.
- [Subhadip Mitra — From 11% to 88% peak bandwidth](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/) — the bandwidth-journey writeup that inspires sub-module 03.

**Hardware-specific reading.**
- [NVIDIA: OpenAI Triton on Blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/) — what Blackwell adds.
- [Colfax: GEMM with TMEM](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) — TMEM and tcgen05, written for CUTLASS but the concepts apply.
- [PyTorch: Enabling vLLM V1 on AMD with Triton](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/) — the AMD parity story.
- [ROCm: Developing Triton Kernels on AMD](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html).

**Production kernels to read.**
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — `src/liger_kernel/ops/*.py`. Start with `rms_norm.py`, then `rope.py`, then `fused_linear_cross_entropy.py`.
- [vLLM `vllm/attention/ops/`](https://github.com/vllm-project/vllm/tree/main/vllm/attention/ops) — the Triton attention backend.
- [SGLang `python/sglang/srt/layers/moe/fused_moe_triton/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe/fused_moe_triton).
- [gpu-mode/triton-index](https://github.com/gpu-mode/triton-index) — a curated catalog of released kernels.

**Tooling.**
- [`triton.proton` profiler docs](https://triton-lang.org/main/profiling/proton.html).
- [Red Hat: Understanding the Triton Cache](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/) (May 2025) — what `~/.triton/cache` is and when to clear it.
- [Red Hat: Triton Kernel Profiling with NVIDIA Nsight Tools](https://next.redhat.com/2025/11/19/triton-kernel-profiling-with-nvidia-nsight-tools/) (Nov 2025).

## Common pitfalls

These eat people. Each is worth its own line in your `notes.md` when you hit it.

1. **You wrote the unfused version but never actually fused.** Check the profiler: the fused kernel should show one HBM read and one HBM write of the right sizes. If you see more, the fusion didn't take.
2. **You forgot `mask=` on `tl.load`.** Loading past the tensor's end gives you garbage from neighboring memory — your kernel "runs" and produces wrong numbers. Always mask boundary loads. Always.
3. **You autotuned without `early_config_prune`.** First autotune of a real kernel will take hours and most configs are illegal. Write the pruning function; it pays back in 10 minutes.
4. **You compared throughputs in different dtypes.** A bf16 kernel at the same GB/s as fp32 is doing twice the useful work. State the dtype in every measurement.
5. **You did not warm up.** First call to a Triton kernel JITs the kernel. `triton.testing.do_bench` defaults are fine; if you roll your own timing, do at least 25 warmup iterations.
6. **You called your timing fast but the kernel never executed.** Cache-key mismatches and grid-size-zero bugs can cause Triton to skip the launch entirely. Always sanity-check the output against eager PyTorch before believing the timer.
7. **Your kernel works on power-of-2 shapes only.** Non-multiple-of-`BLOCK_SIZE` shapes need masks. Test with `H=4097`, not just `H=4096`.
8. **You believe the Liger numbers from their README on a different GPU than yours.** Always re-measure on your hardware. The relative gap is what matters.
