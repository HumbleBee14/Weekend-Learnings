# 05 — GPU Memory Hierarchy

## The mental model

Every level of the memory hierarchy is **5–10× slower** than the one above it. A naive kernel that reads from HBM on every operation pays that 10× tax over and over. A well-structured kernel keeps hot data in shared memory or registers so the HBM round trip is amortized.

This is why fusion matters. This is why FlashAttention exists. This is why GEMMs use shared-memory tiles.

```
Bandwidth ladder (per-SM unless noted; H100 numbers):

  Registers           ~80 TB/s     ──┐
                                     │  ~4× gap
  Shared memory       ~20 TB/s     ──┤
                                     │  ~4× gap
  L2 cache (chip)     ~5 TB/s      ──┤
                                     │  ~1.5× gap
  HBM3 (chip)         3.35 TB/s    ──┘
                                     ↓
                    Cross-GPU NVLink5 ~1.8 TB/s
                                     ↓
                    PCIe 5.0 x16     64 GB/s
                                     ↓
                    Network (InfiniBand) 400 Gb/s = 50 GB/s
```

## Each level in detail

### Registers

The fastest storage. Each thread has its own private set. On H100, 256 32-bit registers per thread (max). Latency ≈ 1 cycle.

Why this matters: **anything you can keep in registers, keep in registers**. The accumulator in a GEMM, running max/sum in a softmax, running output in FlashAttention — all in registers.

The catch: registers are a *finite shared resource per SM*. The SM has 65,536 32-bit registers total. If your kernel uses 128 registers/thread × 256 threads/block × 4 blocks/SM = 131k registers — that won't fit, and either occupancy drops (fewer blocks resident) or values spill to *local memory* (which is actually HBM with a thread-private cache — slow).

Rule of thumb: aim for <128 registers/thread to keep occupancy reasonable. Over 200 is usually a sign your kernel is too fat.

### Shared Memory (SMEM) / L1 cache

A scratchpad shared by all threads in a block. Fast (~30 cycle latency). On H100, 228 KB usable per SM (out of 256 KB combined L1/SMEM).

Three operations:
- Allocate: `__shared__ float buffer[1024]` in CUDA, `tl.zeros((128, 128), tl.float32)` in Triton (compiler decides whether to use SMEM)
- Read/write: like normal arrays, but threads in the same block can see each other's writes after `__syncthreads()`
- Reduce: tree reductions go through SMEM

**Bank conflicts.** SMEM is divided into 32 banks, 4 bytes wide each. If two threads in a warp access different addresses in the *same bank*, they serialize. Visualization:

```
Banks:       0   1   2   3   ...   31
                                          ← warp threads access by column
Address 0:   B0  B1  B2  B3  ...   B31
Address 32:  B0  B1  B2  B3  ...   B31    ← banks repeat every 32 4-byte words
Address 64:  B0  B1  B2  B3  ...   B31
```

If thread 0 reads address 0 and thread 1 reads address 32, both hit bank 0 → 2-way conflict, 2× slower.

The fix: **swizzling** — rearrange the SMEM layout so warp accesses spread across all 32 banks. Triton does this for you. CUTLASS does it via TMA descriptor swizzle modes. Hand-written CUDA C++ has to do it manually (it's a pain).

### L2 cache

Chip-wide (shared by all SMs). 50 MB on H100, 126 MB on B200. ~200-cycle latency, ~5 TB/s bandwidth.

L2 is mostly automatic — you don't allocate from it. But two things matter:

1. **L2 hit rate**. Programs that re-read the same data within a short window get L2 hits → much faster than HBM. The matmul "GROUP_M" trick from Topic 4 exists to improve L2 hit rate.

2. **L2 persistence (`cudaAccessPolicyWindow`)**. CUDA 11+ lets you mark a region of HBM as "preferred for L2 caching." Useful for KV caches in LLM serving.

### HBM (High Bandwidth Memory)

The big pool — 80 GB on H100, 141 GB on H200, 186 GB on B200. ~500-cycle latency, 3.35 TB/s on H100 → 8 TB/s on B200.

Two facts that bite:

1. **Latency vs bandwidth.** HBM has high bandwidth but also high latency. Hiding the latency requires keeping many requests in flight — that's what occupancy is for. Low occupancy → bandwidth utilization drops because the SM stalls waiting on HBM.

2. **Coalescing matters more here than anywhere else.** Uncoalesced HBM accesses can be 5-10× slower than coalesced ones. This is the #1 performance bug in handwritten CUDA.

### Hopper additions: TMA, DSMEM, clusters

Hopper (SM90) added two things that change the bandwidth picture:

**TMA (Tensor Memory Accelerator).** A dedicated copy engine. Instead of having warps issue many small loads, you give the TMA a "descriptor" (a small struct describing a tile) and one async instruction. The hardware moves the tile from HBM → SMEM (or SMEM → SMEM across cluster) without burning warp cycles. Much higher effective HBM bandwidth at low overhead.

```
Pre-Hopper:   warp threads issue 32 separate ld.global → coalesced into bigger transactions by hw
              all 32 threads stalled until load returns
Hopper TMA:   one cp.async.bulk.tensor instruction → hw moves the whole tile
              warp continues running (no stall)
```

**DSMEM (Distributed Shared Memory).** A new tier between SMEM and L2. A *thread block cluster* (group of 2–16 blocks co-resident on the same GPC) shares a unified address space across the participating blocks' SMEM. Threads in block 0 can read/write block 1's SMEM directly, with ~2× the latency of local SMEM but far less than L2.

```
Per-thread:        Registers (256 × 4B)
Per-block:         SMEM (228 KB)
Per-cluster:       DSMEM = cluster_size × 228 KB (up to ~3.6 MB for a 16-block cluster)
Chip-wide:         L2 cache (50 MB on H100)
Chip-wide:         HBM (80 GB on H100)
```

This is what FlashAttention-3 uses to share K/V tiles across CTAs in a cluster instead of re-loading from HBM. Each cluster member loads a slice of K/V into its own SMEM, and other members read those slices via DSMEM. Net effect: K/V is loaded from HBM ~4× less.

### Blackwell additions: TMEM

Blackwell (SM100) added a new on-SM scratchpad called **Tensor Memory** (TMEM). 256 KB per SM. Sits *next to* the tensor cores on-chip.

The new tensor core instruction family `tcgen05.mma` reads operands from SMEM and writes the accumulator into TMEM (not registers). To get the accumulator out, you issue `tcgen05.ld` to copy TMEM → registers.

```
Hopper WGMMA:        SMEM ─┐
                            ├→ Tensor Core → Registers (accumulator)
                     Reg ──┘

Blackwell tcgen05:   SMEM ─┐
                            ├→ Tensor Core → TMEM (accumulator)
                     SMEM ─┘                    │
                                                ▼
                                              Registers (via tcgen05.ld)
```

Why? TMEM has more bandwidth to the tensor cores than registers do, and it doesn't compete with general-purpose register pressure. It's a clean separation between "compute storage" (TMEM) and "general register file."

There's also a "2SM" mode where two adjacent SMs share one MMA — effectively doubling the per-MMA tile size.

For learning purposes: read about TMEM, don't try to write tcgen05 kernels in raw CUDA. Use CuTe-DSL (compiler-and-kernels Level 4) when you need it.

### Multi-GPU: NVLink

NVLink connects GPUs to each other. On Hopper, ~900 GB/s bidirectional per pair. On Blackwell NVLink5, 1.8 TB/s.

NVLink is important when models don't fit on one GPU. Tensor parallelism (TP) splits a matmul across 2/4/8 GPUs and uses NVLink to all-reduce partial sums. This is the regime of Level 6.

### PCIe and the network

Outside the GPU box:
- **PCIe 5.0 x16**: 64 GB/s. The CPU-GPU link. Slow enough that you almost never want to copy tensors over it during inference.
- **InfiniBand / RoCE**: 400 Gb/s = 50 GB/s. Cross-host. Used in distributed training and disaggregated serving.

Each step down is roughly 10× slower than the one above it. The art of multi-GPU training is keeping data inside the fastest tier as long as possible.

## The whole picture (H100 numbers)

```
                 ┌────────────────────────────────────────┐
                 │  REGISTERS    ~80 TB/s  256 × 4B       │  per thread
                 ├────────────────────────────────────────┤
                 │  SMEM         ~20 TB/s  228 KB         │  per block
                 ├────────────────────────────────────────┤
                 │  DSMEM        ~10 TB/s  cluster_size   │  per cluster (Hopper+)
                 │               × 228 KB                 │
                 ├────────────────────────────────────────┤
                 │  L2 CACHE     ~5 TB/s   50 MB          │  chip-wide
                 ├────────────────────────────────────────┤
                 │  HBM3         3.35 TB/s 80 GB          │  chip-wide
                 └────────────────────────────────────────┘
                                   ↓
                 ┌────────────────────────────────────────┐
                 │  NVLink5      1.8 TB/s                 │  GPU ↔ GPU (Blackwell)
                 ├────────────────────────────────────────┤
                 │  PCIe 5.0 x16 64 GB/s                  │  GPU ↔ CPU
                 ├────────────────────────────────────────┤
                 │  IB 400G      50 GB/s                  │  Host ↔ Host
                 └────────────────────────────────────────┘
```

## What this implies for kernel design

| Operation | Where it should live | Why |
|---|---|---|
| Per-thread accumulator | Registers | Read on every iteration; can't afford SMEM latency |
| Tile of A, B in matmul | SMEM | Read many times per output tile; would be slow from HBM |
| Cluster-shared K/V tile (FA3) | DSMEM | Shared across blocks; would re-read from HBM otherwise |
| Model weights | HBM | Too big for SMEM; read once per forward pass |
| KV cache in serving | HBM (with L2 hints for hot prefixes) | Too big; access pattern partly cacheable |
| Cross-GPU tensor partitions | NVLink | Topology dictates this — TP needs NVLink between participating GPUs |

## Bandwidth math you should be able to do

For a forward pass on a 70B parameter model:
- FP16 weights = 140 GB
- HBM bandwidth on H100 = 3.35 TB/s
- Minimum forward time *just to read the weights* = 140/3350 = 42ms

If your decode is taking 100ms/forward, half that is mandatory weight-reading and only the other half is improvable. This is why decode is *memory-bound* and why quantization (smaller weights) helps so directly.

If your decode is 1000ms/forward, the kernel is doing something terribly wrong — there's a 25× gap between achievable and observed.

## 2026 hardware specs (the numbers to memorize)

| GPU | HBM | BW | SMs | SMEM/SM | L2 | Tensor mem | Notes |
|---|---|---|---|---|---|---|---|
| T4 (SM75) | 16 GB GDDR6 | 320 GB/s | 40 | 64 KB | 4 MB | — | Free Colab tier |
| A100 80GB | 80 GB HBM2e | 1.94 TB/s | 108 | 164 KB | 40 MB | — | "the workhorse" |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 132 | 228 KB | 50 MB | — | Hopper baseline |
| H200 | 141 GB HBM3e | 4.89 TB/s | 132 | 228 KB | 50 MB | — | Drop-in for H100 |
| B200 SXM | 186 GB HBM3e | 8 TB/s | 148 | 228 KB | 126 MB | 256 KB | Blackwell |
| GB200 NVL72 | 13.4 TB total | 8 TB/s/GPU | — | — | — | — | 72 B200 + 36 Grace |
| MI300X | 192 GB HBM3 | 5.3 TB/s | 304 CUs | 64 KB | 256 MB | — | AMD CDNA3 |
| Apple M5 Max | up to 128 GB unified | 614 GB/s | 40 GPU cores | unified | unified | — | Mar 2026 launch |

Quick references:
- https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf — H100 architecture whitepaper
- https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor — Blackwell tensor core deep dive
- https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut — B200 microbench
- https://gau-nernst.github.io/tcgen05/ — accessible TMEM / tcgen05 explainer
- https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training — MI300X comparison

## Pitfalls

1. **Believing HBM bandwidth alone determines performance.** Latency matters too — low occupancy → low bandwidth utilization even with the right access pattern. Both axes need attention.
2. **Optimizing for L2 when L1 is the issue.** Profile first. The fix you need depends on which level is the bottleneck.
3. **Putting everything in SMEM.** SMEM is a finite resource per SM. More SMEM/block → fewer blocks resident → lower occupancy. There's a sweet spot.
4. **Treating DSMEM as a free upgrade to SMEM.** It has overhead (atomics across cluster require synchronization). Use it only when the cross-block reuse is significant.
5. **Comparing bandwidths across GPU families without normalizing for FP precision.** A100 BF16 ≠ H100 FP8 ≠ B200 FP4. Always state precision when comparing.
