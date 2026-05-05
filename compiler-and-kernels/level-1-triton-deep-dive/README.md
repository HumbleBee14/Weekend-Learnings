# Level 1 — Triton Deep Dive

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: fused RMSNorm+RoPE kernel with warp specialization

## Week goal

You used Triton in `systems-for-ml` Level 2 to write a basic matmul. This week you go much deeper — the advanced patterns that production teams (vLLM, SGLang, FlashInfer, Liger-Kernel) actually use. By Friday you should be able to:

- Write a **persistent kernel** that occupies all SMs and avoids kernel re-launch overhead
- Apply **warp specialization** so producer warps prefetch via TMA while consumer warps run tensor core compute
- Use **autotune** properly — not just `@triton.autotune` with a list of configs, but with custom pruning to avoid wasteful trials
- Write a **fused RMSNorm+RoPE kernel** that beats Liger-Kernel's reference (or matches it and understands why)
- Run your kernel on **AMD ROCm** via the same Triton code and measure the portability gap

## Where this fits

- **Comes after:** `systems-for-ml` Level 2 (you wrote a basic matmul in Triton; you know what `triton.language` is) and Level 3 (you can read a profiler trace and place a kernel on the roofline).
- **Comes before:** Level 2 of this track (`torch.compile` internals — Inductor emits Triton, so being fluent in Triton makes the Inductor output readable).

## 2026 reality check

- **Triton 3.2 ships with PyTorch 2.6** and includes automated warp specialization via a compiler pass called Tawa (arxiv 2510.14719). You can write a warp-specialized kernel manually today, or annotate with `@triton.jit(launch_cooperative_grid=True)` and let the pass do it. Understanding the manual version first is the right order.
- **TMA (Tensor Memory Accelerator)** is the Hopper instruction that enables async loads to shared memory without warp overhead. In Triton it's exposed as `tl.make_tensor_descriptor` + `tl.load`. It's what makes FA3 fast and what Tawa uses internally.
- **Liger-Kernel** (LinkedIn, MIT license) is the production reference for fused elementwise ops: RMSNorm, RoPE, SwiGLU, cross-entropy. Their implementations achieve 7–8× speedup and 3× memory reduction vs eager PyTorch. Your goal this week is to match them and understand every decision they made.
- **AMD ROCm** runs the same Triton code. NVIDIA A100/H100 is the primary target, but Triton 3.3 ships with ROCm 7.0 for MI300X. The portability is real — with some caveats around TMA (Hopper-only) and tensor core instructions.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | triton-execution-model | Grid, blocks, warps in Triton vs raw CUDA — the mapping |
| 02 | persistent-kernels | Occupy all SMs for the full problem; avoid launch overhead |
| 03 | warp-specialization | Producer/consumer split; TMA + WGMMA ping-pong |
| 04 | advanced-autotune | Config pruning, `early_config_prune`, proton profiler |
| 05 | fused-rmsnorm-rope | Write it, benchmark it, read Liger-Kernel's version |
| 06 | split-k-and-reductions | Parallel reductions across tiles with atomic ops |
| 07 | triton-on-amd | Same kernel on ROCm MI300; measure portability cost |

### 01 — `triton-execution-model`

**The mapping from Triton to hardware.** In Triton, a `@triton.jit` function launches a 1D or 2D grid of *programs* (the Triton word for "thread blocks"). Inside each program you operate on *tiles* — contiguous blocks of data you specify with `tl.arange` and `tl.load`. The compiler maps these tiles to warp groups, schedules shared memory allocation, and emits the PTX. Unlike CUDA, you never specify threads or shared memory explicitly — Triton infers them from the tile sizes.

**Key conceptual shift.** In CUDA, you think "one thread per output element." In Triton, you think "one program (block) per output tile." The compiler handles vectorization and the thread-level details. This is why Triton scales better to different GPUs — the tile-to-hardware mapping is the compiler's problem, not yours.

**What to read.** Re-read the Triton matmul tutorial with this framing in mind. Then read the `tl.load` / `tl.store` docs and understand the `mask` parameter — it's how Triton handles boundary conditions that CUDA handles with if-guards.

### 02 — `persistent-kernels`

**What non-persistent kernels do wrong.** A standard GEMM kernel launches a grid of `(M/BLOCK_M) × (N/BLOCK_N)` programs. The GPU schedules them in waves: each SM picks up a program, runs it, picks up another. If the problem doesn't fill the GPU (small batch sizes, common in LLM decode), the last few programs run on a small number of SMs with the rest idle. Worse, each kernel launch has overhead — the hardware scheduler, L2 cache warm-up, register file refill.

**What persistent kernels do.** Launch exactly `num_SMs` programs (one per SM). Each program loops over multiple output tiles internally. This eliminates launch overhead and allows the SM to prefetch the next tile's data while computing the current tile. Combined with warp specialization, it becomes the "doubly-nested loop" pattern.

**Build steps.**
1. Start from a standard Triton matmul.
2. Add a persistent outer loop: `tile_id = tl.program_id(0)` iterates over a flat tile space, each program handles `ceil(total_tiles / num_SMs)` tiles.
3. Benchmark: persistent vs standard on a sequence of small GEMMs (batch decode shapes: M=1, M=8, M=16). The persistent version should win because of reduced launch overhead.
4. Profile with `triton.proton` — compare SM occupancy and L2 hit rate between the two.

### 03 — `warp-specialization`

**The idea.** Warps within a threadblock are split into two groups: *producers* that run TMA loads (async memory → shared memory) and *consumers* that run WGMMA (shared memory → tensor cores → register). They run simultaneously — consumers can start computing while producers are still loading the next tile.

**Why this matters.** Without warp specialization, the entire warp stalls waiting for a memory load. With it, the GPU can keep tensor cores busy at near-peak utilization. FA3's 1.5–2× speedup over FA2 comes almost entirely from this pattern.

**In Triton.** Triton 3.2's `@triton.jit(launch_cooperative_grid=True)` combined with `tl.make_tensor_descriptor` + `tl.async_copy` enables the TMA path. The Tawa compiler pass then automates producer/consumer assignment. To learn the internals, implement it manually first — split the warp groups explicitly using `tl.num_warps` and `tl.warp_id`, then let the compiler take over.

**Resources.**
- [PyTorch warp specialization blog](https://pytorch.org/blog/warp-specialization/)
- [Tawa paper — arxiv 2510.14719](https://arxiv.org/html/2510.14719)
- [Triton FA3 warp-spec PR #5622](https://github.com/triton-lang/triton/pull/5622)

### 04 — `advanced-autotune`

**Why naive autotune wastes time.** `@triton.autotune` with a flat list of configs tries all combinations: `BLOCK_M × BLOCK_N × BLOCK_K × num_warps × num_stages`. On an H100 this can be 200+ configs. Many are nonsensical (BLOCK_M=256 with num_warps=1 doesn't fit in registers). Naive autotune runs all of them — slow and noisy.

**`early_config_prune`** lets you write a Python function that discards configs before benchmarking. You check register pressure (`num_warps * 32 * BLOCK_M * BLOCK_K * 4 bytes ≤ SMEM_per_SM`), occupancy constraints, and hardware-specific rules. This cuts config space by 70–80%.

**`triton.proton`** is Triton's built-in kernel profiler (shipped with Triton 3.x). It provides per-kernel hardware counters without needing Nsight. Use it to understand which config won and why — not just which was fastest.

**Build steps.**
1. Take your matmul. Write an `early_config_prune` that filters configs exceeding SMEM budget.
2. Profile the winning config with `triton.proton`. Look at: achieved TFLOPS, SMEM bandwidth utilization, L2 hit rate.
3. Compare to `torch.compile`'s Inductor-generated Triton for the same shape — Inductor runs its own autotune. Which wins and why?

### 05 — `fused-rmsnorm-rope`

**Why this kernel matters.** In a LLaMA-style model, every single token passes through RMSNorm + RoPE twice per layer (pre-attention and post-attention). Unfused, this is four separate CUDA kernels, four HBM round-trips. Fused, it's one kernel and one round-trip. For memory-bound decode workloads this is a direct TTFT + throughput improvement.

**RMSNorm.** Computes `x / sqrt(mean(x²) + ε) * weight`. Requires a reduction over the hidden dimension (256–8192 elements), then an elementwise scale. The reduction is the hard part — it's a two-pass algorithm (compute mean, then normalize) unless you use online normalization.

**RoPE (Rotary Position Embedding).** Applies a rotation matrix to pairs of hidden dimensions based on position. Elementwise per token, no inter-token communication. Very fast, very memory-bandwidth-bound.

**Build steps.**
1. Write RMSNorm in Triton. One program per row (token). Use a parallel reduction across `BLOCK_SIZE` elements to compute the mean.
2. Write RoPE in Triton.
3. Fuse them: a single program reads the input once, computes RMSNorm reduction, applies normalization, then applies RoPE, and writes the output once.
4. Benchmark vs eager PyTorch, vs `torch.compile`, vs [Liger-Kernel's implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py). Measure: throughput (tokens/sec), peak memory bandwidth utilization (GB/s vs GPU peak), memory saved.
5. Profile with `triton.proton` or `nsys`. Confirm: one DRAM load, one DRAM store.

### 06 — `split-k-and-reductions`

**When split-K matters.** For GEMM with a large K dimension (long sequences, large hidden dims) but small M (batch=1 decode), the standard tiling leaves most of the K dimension sequential. Split-K partitions K across multiple SMs — each SM computes a partial sum, then atomic-adds to the output. This trades latency for parallelism on small-M problems.

**`tl.atomic_add`.** Triton's atomic add for partial reductions across programs. The pattern: each program writes to a scratchpad, a reduce kernel combines them. For simple cases, `tl.atomic_add` with `sem="relaxed"` is enough.

**When not to use it.** On large batches split-K hurts (extra synchronization cost outweighs the parallelism benefit). Know the crossover point for your shape.

### 07 — `triton-on-amd`

**What works without changes.** Pure Triton code (no TMA, no WGMMA) runs on AMD ROCm MI300X via the same `.py` file. The Triton ROCm backend targets `gfx940` (MI300X) and `gfx942` (MI308X). `triton.autotune` still works; the auto-tuner picks different optimal configs for AMD vs NVIDIA hardware.

**What needs changes.**
- TMA (`tl.make_tensor_descriptor`) is Hopper-only. AMD equivalent is not yet in Triton ROCm backend (2026). Skip it or guard with `if tl.constexpr(HAS_TMA)`.
- `num_stages` defaults differ — AMD MI300X has a different shared memory capacity and latency profile. The autotuner handles this but starting configs should differ.
- `BLOCK_SIZE` optimal values differ — AMD's memory bus width is 4096 bits vs NVIDIA's 1024 bits per memory partition. Wider blocks benefit more on AMD.

**Build steps.**
1. Take your fused RMSNorm+RoPE kernel.
2. Run it on a RunPod MI300X instance (available ~$2/hr on Vast.ai or RunPod).
3. Note what `triton.autotune` picks for AMD vs NVIDIA — the winning config will differ.
4. Measure: AMD MI300X throughput (GB/s) vs NVIDIA A100 throughput (GB/s). Both chips have similar HBM bandwidth (~2 TB/s); the gap should be small on memory-bandwidth-bound ops.

## Project this week

```
compiler-and-kernels/
└── kernels/
    ├── rmsnorm_rope_fused.py     # your kernel
    ├── benchmark.py              # vs eager, torch.compile, Liger-Kernel
    └── reports/
        └── level1-triton.md     # profiler screenshots, roofline placement, AMD vs NVIDIA numbers
```

**The benchmark table you should produce:**

| Kernel | Throughput (GB/s) | vs HBM peak (%) | Memory saved |
|---|---|---|---|
| Eager PyTorch (unfused) | | | baseline |
| torch.compile | | | |
| Liger-Kernel | | | |
| Your fused (no warp spec) | | | |
| Your fused (warp spec, H100 only) | | | |

If your kernel matches Liger-Kernel's bandwidth utilization, you've done it right. If it beats it, read the diff — either you found something, or you measured wrong.

## Definition of done

- [ ] You have a working fused RMSNorm+RoPE Triton kernel with benchmark numbers vs Liger-Kernel.
- [ ] You understand what a persistent kernel is and when it beats standard grid launch.
- [ ] You can explain warp specialization — producer/consumer split, TMA, the ping-pong pattern.
- [ ] You ran your kernel on AMD (or understand what changes would be needed and why).
- [ ] Your `reports/level1-triton.md` has the benchmark table, a profiler screenshot, and roofline placement.

## Resources

- **Triton docs** — [triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/index.html). Tutorial 06 (fused attention) is the most relevant this week.
- **Liger-Kernel** — [github.com/linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel). Read `ops/rms_norm.py` and `ops/rope.py` carefully.
- **Anatomy of a Triton Attention Kernel** — [arxiv.org/abs/2511.11581](https://arxiv.org/html/2511.11581v1). Maps every Triton construct to hardware.
- **Tawa paper** — [arxiv.org/abs/2510.14719](https://arxiv.org/html/2510.14719). The warp specialization compiler pass.
- **triton.proton docs** — [triton-lang.org/main/profiling/proton](https://triton-lang.org/main/profiling/proton.html).
- **AMD ROCm Triton guide** — [rocm.blogs.amd.com/artificial-intelligence/triton](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html).
- **NVIDIA Triton on Blackwell blog** — [developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/).

## Common pitfalls

1. **Writing the unfused version first but never actually fusing.** The fusion is the point — one read, one write, no intermediate HBM traffic. Verify it in the profiler.
2. **Ignoring boundary conditions.** If your hidden dim isn't a multiple of `BLOCK_SIZE`, `tl.load` with a mask is required. Missing this gives silent wrong answers on non-power-of-2 shapes.
3. **Trusting autotune without warmup.** The first autotuned trial includes kernel JIT. Always include `warmup=25` in `triton.testing.do_bench`.
4. **Comparing throughput without fixing dtype.** BF16 and FP32 kernels at the same GB/s are very different — BF16 moves half the bytes for the same work. State the dtype in every benchmark.
5. **Skipping the AMD test.** The portability test is where you learn what's hardware-specific in your kernel vs what's truly portable.

## What you'll be able to do after this week

> Write production-quality Triton kernels with warp specialization and persistent grid patterns. Profile them correctly, benchmark against established references like Liger-Kernel, and port them to AMD ROCm. Understand exactly why kernel fusion eliminates HBM round-trips and how to measure that elimination in a profiler.
