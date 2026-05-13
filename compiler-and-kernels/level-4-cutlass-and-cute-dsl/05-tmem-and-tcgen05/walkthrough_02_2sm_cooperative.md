# Walkthrough 02 — 2-SM cooperative MMA

> Source: [`cutlass/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu)

The 2-SM cooperative pattern is what unlocks Blackwell's peak BF16/FP8/FP4 throughput. Two CTAs in the same cluster cooperatively run one MMA tile. Each carries half the operands and half the accumulator. The leader issues; the peer contributes.

## What the kernel does

Same operation as walkthrough 01 (`C = A @ B`), but with cluster `(2,1,1)`. The MMA tile is now `m256n256k16` — twice the M of single-SM. The leader CTA does `m=[0..128)`; the peer does `m=[128..256)`. Each holds its 128 rows of the accumulator in its own TMEM.

## The new pieces

### Cluster setup

```cpp
constexpr Shape<_2, _1, _1> ClusterShape{};
// launch with cute::dim3{2,1,1} as cluster
// Inside the kernel:
int cta_rank_in_cluster = cute::block_rank_in_cluster();   // 0 or 1
bool is_leader = (cta_rank_in_cluster == 0);
```

### TMA multicast

A single `SM90_TMA_LOAD_MULTICAST` instruction with a 2-bit cluster mask `0b11` lands the operand in *both* CTAs' SMEM. You don't issue two separate TMAs; one TMA, both SMEMs filled.

```cpp
// Load A: each CTA gets its half-of-M slice (different data per CTA)
copy(tma_a_per_cta, gA_my_half, sA);

// Load B: both CTAs need the same N slice (same data, multicast)
copy(tma_b_multicast, gB_shared, sB);  // multicast mask = cluster_mask
```

The B operand is multicast because both CTAs in the pair share the same N range. The A operand is partitioned because each CTA owns a different M range.

### The MMA atom

```cpp
TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_2SM_SS{});
//                                                      ^^^^^
//                                                  2-SM variant
```

The `_2SM_` in the atom name lowers to `tcgen05.mma.cta_group::2`. The TMEM accumulator handle is the same shape per CTA (128 rows), but the MMA logically spans both.

### The leader-only issue

```cpp
cute::elect_one_sync();                   // pick one CTA in pair (the leader)
if (is_leader && elect_one_thread()) {
  gemm(tiled_mma, tCrA, tCrB, tCtAcc);    // tcgen05.mma.cta_group::2
}
```

Only the leader's one thread issues. The peer waits — its TMEM is being written by the leader's MMA. Both CTAs proceed in sync via the cluster barrier.

### The per-CTA epilogue

Each CTA writes its half. The output tile in GMEM is `(M_TILE=256, N_TILE=256)`; CTA rank 0 writes rows `[0..128)`, CTA rank 1 writes rows `[128..256)`. Each does its own `tcgen05.ld` from its own TMEM.

## Things to internalize

**Cluster is at least 2 for 2-SM MMA.** You can't 2-SM with a cluster of 1. The cluster size *is* the cooperation group size.

**Leader vs peer is by `block_rank_in_cluster`.** Rank 0 is leader. Rank 1 is peer. Higher ranks would matter for cluster-of-4 (which exists but is rare in BF16; common in FP4).

**TMA multicast vs partitioned is a per-operand choice.** A operand: each CTA needs a different slice → partitioned, separate TMA descriptors with offset arithmetic per CTA rank. B operand: both CTAs need the same data → multicast TMA, one descriptor, one instruction.

**The leader's MMA writes both CTAs' TMEM.** Sounds like magic; it's not — the hardware in the cluster has the cross-CTA TMEM bus. Conceptually the MMA writes to a "logical" `(256, 256)` accumulator that physically lives split across the pair.

**Synchronization is cluster-aware.** Instead of `__syncthreads()` (CTA-local) you use `cluster_arrive_relaxed` / `cluster_wait` for cluster-wide sync. Mbarriers in cluster shared memory support multi-CTA arrivals.

## Why this matters

The B200 BF16 tensor-core peak is ~4.5 PFLOPS sparse / ~2.25 PFLOPS dense. To approach that you need MMA tiles big enough to saturate. Single-SM MMA peaks around `m128n256k16`. 2-SM MMA goes to `m256n256k16` — twice the arithmetic per issue. On large square GEMMs the 2-SM path is 1.5–1.8× the single-SM path.

For decode-shape GEMMs (small M), 2-SM is a loss — you can't keep both CTAs busy. The cluster shape is a tuning knob you set per-shape.
