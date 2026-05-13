# 05 — TMEM and tcgen05 on Blackwell (SM100)

> Outer: [`../README.md`](../README.md) · Hardware: B200 optional. Without it, this is a read-along.

Blackwell's tensor cores changed three things that the kernel author must understand even if they never write a Blackwell kernel directly: the accumulator moved out of registers into TMEM, the MMA is issued by a single thread, and pairs of CTAs can cooperatively run one MMA tile. FlashAttention-4 uses all three. The next two years of NVIDIA kernel work will too.

If you have a B200, you can port submodule 04's SM90 kernel to SM100 and measure. If you don't, you walk through three annotated files and finish with a precise mental model.

## What changed, and why each change exists

### 1. The accumulator lives in TMEM, not registers

**Hardware change.** Blackwell adds Tensor Memory: 256 KB per SM, organized as 128 lanes × 512 columns of 32-bit cells. It sits adjacent to the tensor cores. The MMA result writes *here*, not into the register file.

**Why.** The Blackwell tensor cores got bigger. The native MMA tile is `m128n256k16` for BF16/FP16. An accumulator that big in FP32 is `128 × 256 × 4 = 128 KB` — more than the register file per warpgroup can hold without spilling. TMEM provides bigger storage that's still tensor-core-adjacent.

**Consequence for kernel authors.** You explicitly `tcgen05.alloc` columns of TMEM at the start of the kernel, the MMA writes its result there, and you `tcgen05.ld` it into registers for the epilogue. The deallocation (`tcgen05.dealloc`) is mandatory before kernel exit — forgetting it hangs subsequent launches.

**Addressing.** A TMEM address is `(lane_id << 16) | column`. So column 0 of lane 5 is `0x00050000`. The high 16 bits are the lane (0..127); the low 16 bits are the column. Allocation is column-granular; allocate 256 columns, you get a 128-row × 256-column slab.

### 2. tcgen05.mma is issued by one thread

**Hardware change.** WGMMA on Hopper was issued by a full warpgroup (128 threads, all 4 warps synchronizing). `tcgen05.mma` on Blackwell is issued by *one thread* on behalf of the entire CTA.

**Why.** The single-thread issue simplifies the kernel structure — no warpgroup-wide barrier at the issue site. The hardware does the heavy lifting; the kernel just tells it where the inputs are.

**Consequence.** You typically have one warp ("MMA warp") whose sole job is to issue `tcgen05.mma` instructions. Other warps load tiles and consume results. The five-warp specialization in FA4 is built on this — each warp has a narrow role because the MMA only needs one thread to drive it.

**But.** Reading the *result* out of TMEM is warpgroup-wide. `tcgen05.ld` is a warpgroup instruction; one warp can only see 32 of the 128 TMEM lanes (the lanes assigned to its warp ID). So the epilogue still needs a full warpgroup.

The FA4 kernel uses this asymmetry directly — five warp roles, deliberately uneven in width:

```
Warp role            Threads  What it does                       Width
──────────────────  ────────  ────────────────────────────────  ────────
Load   warp           32      issue TMA for Q, K, V tiles       1 warp
MMA    warp            1*     issue tcgen05.mma instructions    1 thread
Softmax warp          32      online-softmax stats              1 warp
Correction warp       32      rescale prev block when m updates 1 warp
Epilogue warpgroup   128      tcgen05.ld + write O to GMEM      4 warps

* The MMA warp exists as a full warp for scheduling, but only one
  elected thread issues each tcgen05.mma. The other 31 wait.
```

*Each role is sized to what the hardware actually needs — MMA wants one thread, TMEM readback wants a full warpgroup.*

### 3. 2-SM cooperative MMA

**Hardware change.** Two CTAs in the same cluster — a "CTA pair" — can cooperatively execute one MMA tile. The MMA is `tcgen05.mma.cta_group::2`. Each CTA loads half the A and B operands and holds half the accumulator in its TMEM. The "leader" CTA in the pair issues the MMA instruction; the "peer" CTA contributes its operand and TMEM but doesn't issue.

**Why.** Single-SM MMA can't saturate the bigger tensor cores at full bandwidth. With 2-SM cooperation you push twice as much arithmetic per MMA, hitting peak throughput.

**Consequence.** Your CTA layout becomes cluster-aware. The cluster size is at least 2 (often `(2,1,1)` or `(2,2,1)`). TMA loads use cluster-multicast (`SM90_TMA_LOAD_MULTICAST` analog) so both CTAs in the pair get their operand half with one load instruction.

## Three annotated walkthroughs

The folder contains three commentary files. Open each alongside the linked CUTLASS source.

### A — `walkthrough_01_single_sm_mma.md`

Pairs with [`cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu).

The minimal single-SM `tcgen05.mma` kernel. Annotates:
- `tcgen05.alloc` — column allocation, the base TMEM pointer
- The MMA atom (`SM100_MMA_F16BF16_SS`) and how its accumulator layout has the `(lane << 16 | col)` address structure
- The single-thread issue (`if elect_one_warp`)
- `umma_arrive` — the MMA's completion signal to a barrier
- `make_tmem_copy` and `tcgen05.ld` — the warpgroup-wide TMEM→register move for the epilogue
- `tcgen05.dealloc` before kernel exit

### B — `walkthrough_02_2sm_cooperative.md`

Pairs with [`cutlass/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu).

The 2-SM cooperative MMA. Annotates:
- Cluster shape `(2,1,1)` and how `block_rank_in_cluster` identifies the leader (rank 0) vs peer (rank 1)
- TMA multicast: one `SM90_TMA_LOAD_MULTICAST` operation lands the operand in *both* CTAs' SMEM
- `tcgen05.mma.cta_group::2` issued only by the leader's elected thread
- How each CTA's TMEM holds half the accumulator
- Why the epilogue is per-CTA — each writes its half of the output tile

### C — `walkthrough_03_persistent_dense_gemm_sm100.md`

Pairs with [`cutlass/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py).

The Python equivalent of submodule 04's stage 5, but on SM100. Annotates the deltas from the SM90 version:
- The MMA atom swap (`SM90_64x128x16_F32BF16BF16_SS` → `SM100_MMA_F16BF16_SS`)
- TMEM allocation in the kernel prologue
- The accumulator handle being a TMEM pointer instead of a register fragment
- The epilogue's `tcgen05.ld` to bring the FP32 accumulator into registers for cast-and-store
- The cluster setup if 2-SM is enabled

## Hands-on (B200 only)

If you have B200 access:

1. Run the upstream `blackwell/dense_gemm_persistent.py` at M=N=K=4096, BF16. Should land >80% of cuBLAS BF16.
2. Take your `stage5_persistent.py` from submodule 04 and port it: swap MMA atom, add TMEM allocate/dealloc, change accumulator handling. Benchmark.
3. Optional: enable 2-SM cooperative MMA. Cluster (2,1,1); TMA multicast for B. Measure the gap.

If you don't have B200, you finish the walkthroughs and write a one-page summary in `notes.md` of how the Blackwell kernel differs from the Hopper one. The capstone's NVFP4 section depends on you understanding this submodule.

## What "Triton can't fully exploit this yet" actually means

Triton has been adding Blackwell support. `tl.dot` lowers to `tcgen05.mma` on SM100. But two features are not yet exposed cleanly in Triton:

1. **TMEM-resident accumulator pipelining.** Triton manages the accumulator. You can't tell it "keep the accumulator in TMEM across multiple K passes while you do other work in registers." FA4 does this; it's the basis of the correction-warp pattern.
2. **2-SM cooperative MMA with explicit cluster control.** Triton can launch with clusters but doesn't expose CTA-rank-conditioned MMA issue. The `cta_group::2` variant is the one that hits peak.

The migration path: Gluon (Triton's lower-level dialect) is adding these. CuTe-DSL exposes them today.

## What you should be able to do next

- Read FA4's Blackwell kernel and identify the TMEM allocation, the MMA warp role, the correction warp role, the `tcgen05.ld` in the epilogue.
- Predict the SM100 version of a Hopper kernel — what changes, what stays.
- Explain to a colleague why `tcgen05.mma` is issued by one thread but `tcgen05.ld` is warpgroup-wide.
- Argue convincingly for or against using 2-SM cooperative MMA on a given shape.

## References

- [Colfax: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) — canonical tutorial.
- [Colfax: GEMM with Thread Block Clusters on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) — 2-SM cooperative pattern.
- [Colfax: Hardware-supported Block-scaling on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/) — NVFP4/MXFP block scaling.
- [Colfax: Sub-byte GEMM on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/) — FP4 and FP6.
- [gau-nernst: tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) — plain CUDA, 98% of cuBLAS at 4096³.
- [SemiAnalysis: Dissecting Nvidia Blackwell](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor) — the hardware in detail.
- [Microbenchmarking Blackwell (arXiv 2512.02189)](https://arxiv.org/html/2512.02189v1) — measured numbers.
- [Modal: We reverse-engineered FA4](https://modal.com/blog/reverse-engineer-flash-attention-4) — how all of this is actually used.
- [NVIDIA: Blackwell SM100 Functionality docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).
