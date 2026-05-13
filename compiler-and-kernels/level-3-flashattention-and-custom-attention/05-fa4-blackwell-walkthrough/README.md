# 05 — FA4 on Blackwell: a read-only walkthrough

> Prereq: sub-modules 03–04. Hardware: B200 optional; this is a read + write sub-module.

FA4 ([Zadouri, Bikshandi, Dao et al, arXiv 2603.05451, Mar 5 2026](https://arxiv.org/abs/2603.05451)) is FlashAttention's first non-trivial re-architecture since FA2 — every prior version applied the same online-softmax-tiling algorithm with different scheduling. FA4 keeps that algorithm but rebuilds the kernel around Blackwell's "asymmetric hardware scaling": tensor core throughput doubled vs Hopper, but SFU and shared-memory bandwidth grew much less. The result: the non-matmul path (exp, rescale) became the new bottleneck. FA4 attacks it.

Performance: on B200 BF16, ~1605 TFLOPs/s (71% of peak), about 1.3× cuDNN 9.13 and 2.7× a tuned Triton attention kernel. On Hopper, FA3 is still the right answer.

You do not implement FA4. You read it. The spine of the walkthrough is [Modal's "We reverse-engineered Flash Attention 4"](https://modal.com/blog/reverse-engineer-flash-attention-4) blog plus the FA4 paper plus [Colfax's FA4 post](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/).

## The five changes that matter

### 1. Five warp specializations, not two

FA3 had producer/consumer with ping-pong (effectively 3 roles: load, MMA-group-A, MMA-group-B). FA4 explodes this:

- **Load warps** (1): issue TMA for K and V.
- **MMA warps** (1): issue WGMMA / tcgen05 MMA for QK^T and PV.
- **Softmax warps** (8): compute row max, exponentials, row sum, rescale factors.
- **Correction warps** (4): apply the rescale to O.
- **Epilogue warps** (1–2): write O to HBM.

That's ~16 roles versus FA3's 3. Why so many? Blackwell's async MMA can have more outstanding ops in flight. To keep the tensor cores fed you need more concurrent work *off* the tensor cores — meaning more warps dedicated to softmax and correction. The schedule is more like a CPU instruction pipeline than a GPU loop.

This is why FA4 is in CuTe-DSL, not Triton. Triton's `warp_specialize=True` is good for "split into producer and consumer." For five-way custom partitioning with hand-tuned barrier patterns, you need lower-level control.

### 2. Software-emulated exponentials

The single most important Blackwell-specific change. On Hopper, FA3 routed `exp` through the SFU via the PTX `exp2.approx.ftz.f32` instruction — about 30% of consumer-warp time. On Blackwell, the FMA-to-SFU ratio roughly doubled, so SFU `exp` is even more bottleneck-shaped relative to the matmul path.

FA4 approximates `2^x` on the unit interval with a **cubic polynomial in pure FMAs**. The output matches the SFU result to bf16 precision (well within attention's tolerance). The polynomial is applied selectively — small head dims (d=64) benefit most because the softmax work is a larger fraction of the per-tile cost.

This is the lesson the kernel writer should internalize: **when a hardware unit doubles in throughput but its neighbors don't, kernels rebalance away from the units that didn't scale.** Blackwell didn't bump SFU; FA4 stops using it.

### 3. Conditional softmax rescaling

The rescale step `O = O * exp(m_old - m_new)` is a no-op when `m_old == m_new` (exp(0) = 1). FA3 still issues the FMAs because checking the condition costs a branch.

FA4 tracks per-row "max changed this tile" flags and **skips the rescale FMAs** when the max didn't change. On typical attention rows the running max stabilizes early — after the first few tiles, ~90% of subsequent tiles don't update it. Tri Dao reported ~10× fewer correction ops on representative prompts. With Correction warps as a dedicated role, the savings cascade: fewer FMAs means those warps can spend their cycles on softmax instead.

This is the property you derived in sub-module 02 — when `m_new == m_old`, rescale is `1.0` and free. FA4 makes "free" actually free by eliminating the FMA issue, not just the no-op result.

### 4. 2-CTA cooperative MMA

Blackwell added a mode where **two SMs can cooperate on a single MMA instruction** (`tcgen05.mma.cta_group::2` in PTX). The two SMs share K and V loads (one SM loads each), halving the per-SM SMEM pressure and effectively giving each SM access to 2× the SMEM for the same tile size. FA4 uses this for the backward pass to keep `dK, dV` partial sums in tensor memory instead of HBM round-tripping through atomics.

Triton does not yet expose 2-CTA MMA. FA4 uses inline PTX. This is one of the gaps that motivates Level 4 (CuTe-DSL): the lowest-level control over the hardware that doesn't require writing CUDA C++.

### 5. CuTe-DSL, not C++

FA3's source is ~70k lines of CUDA C++ templates in CUTLASS style. FA4's forward kernel is ~2000 lines of Python with `@cute.jit`. Same hardware control, dramatically less code, 20–30× faster compile times. The DSL exposes CuTe (the CUTLASS layout algebra) directly in Python, plus inline PTX for the bits CuTe doesn't model yet.

The pedagogical reason this matters for you: FA4 is the first major production kernel that **a researcher with a kernel idea can read end-to-end in an afternoon**. Compare to FA3's CUTLASS templates, which take weeks to navigate. The democratization of kernel writing is a real thing, and CuTe-DSL (Level 4) is the tool that enables it.

## What you build

A single artifact: a one-page write-up in `notes.md` explaining each of the five changes above to a teammate who has finished sub-module 04 (so they know FA3). For each change, answer:

- What problem did it solve?
- What hardware feature made the solution possible?
- Why couldn't this have been done in FA3?

If you have a B200, also: install FA4 (`pip install flash-attn-cute` or build from source per the FA4 README), run the provided benchmark script `bench_fa4_if_blackwell.py`, paste numbers. Bit-verify against the FA2 NumPy reference from sub-module 03 on a small shape to confirm the kernels agree algorithmically.

## What you do *not* build

A working FA4 kernel. Tri Dao's team took months. The point of this sub-module is *to read*, not to reproduce. If you find yourself trying to reproduce FA4, redirect that energy to Level 4 (CuTe-DSL) where you build up the tools properly.

## Definition of done

- [ ] One-page write-up in `notes.md` covering the five changes.
- [ ] You can explain to a teammate why FA4 is CuTe-DSL and not Triton.
- [ ] You can explain why FA4 is Blackwell-only and not a portable improvement.
- [ ] (Optional, B200 only) Benchmark numbers from `bench_fa4_if_blackwell.py`.

## References

- [FA4 paper — arXiv 2603.05451](https://arxiv.org/abs/2603.05451) (Mar 5, 2026). Sections 2 (background) and 3 (algorithm) are the core.
- [Modal — We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4). Better entry point than the paper for the first read.
- [Colfax Research — FA4 algorithm and kernel pipelining](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/). The most technical companion.
- [Lambda — FA4 announcement](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet).
- [Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention) `cute_dsl/` directory — the actual source. Recommended after the Modal blog.
- [NVIDIA CuTe-DSL docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) — for Level 4 next.
