# 01 — CUDA Mental Model

## Why this exists

Before any code: get the execution model right. Most "why is my kernel slow?" answers are one of: warp divergence, low occupancy, bank conflicts, memory bandwidth saturation. None of them make sense without the mental model.

This is foundation. Read it twice if you need to.

## What a GPU actually is

A GPU is not "a really fast CPU." It's a **throughput machine** — thousands of small, simple cores that only go fast when you give them lots of similar work to do.

CPU analogy (since it helps to anchor): a CPU is a few highly skilled chefs in a kitchen, each doing complex work in parallel. A GPU is a giant assembly line of 10,000 workers, each doing one tiny step very fast — but only if they're all doing the *same* step.

The catch: those 10,000 workers are organized into groups of 32 that *must* execute the same instruction at the same time. If one of them branches differently, the others wait. That's the SIMT model — Single Instruction, Multiple Threads.

## The execution hierarchy

```
                 ┌─────────────────────────────────────────────────┐
                 │                  GRID                           │
                 │  (one kernel launch — the whole problem)        │
                 │                                                 │
                 │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
                 │  │ BLOCK 0 │  │ BLOCK 1 │  │ BLOCK 2 │  …       │
                 │  │         │  │         │  │         │          │
                 │  │ ┌─────┐ │  │ ┌─────┐ │  │         │          │
                 │  │ │ W0  │ │  │ │ W0  │ │  │         │          │
                 │  │ │ 32  │ │  │ │ 32  │ │  │         │          │
                 │  │ │thrds│ │  │ │thrds│ │  │         │          │
                 │  │ └─────┘ │  │ └─────┘ │  │   …     │          │
                 │  │ ┌─────┐ │  │ ┌─────┐ │  │         │          │
                 │  │ │ W1  │ │  │ │ W1  │ │  │         │          │
                 │  │ └─────┘ │  │ └─────┘ │  │         │          │
                 │  │   ...   │  │   ...   │  │         │          │
                 │  └─────────┘  └─────────┘  └─────────┘          │
                 └─────────────────────────────────────────────────┘
                              ↓ scheduled onto ↓
                 ┌─────────────────────────────────────────────────┐
                 │   SMs (Streaming Multiprocessors) — physical    │
                 │   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐    …       │
                 │   │ SM0 │  │ SM1 │  │ SM2 │  │ SM3 │            │
                 │   └─────┘  └─────┘  └─────┘  └─────┘            │
                 │   H100: 132 SMs   B200: 148 SMs                 │
                 └─────────────────────────────────────────────────┘
```

**Levels from largest to smallest:**

- **Grid** — one whole kernel launch. You decide its shape: `<<<grid_dim, block_dim>>>`.
- **Block** (also called CTA — Cooperative Thread Array) — a group of threads that runs together on *one* SM. Threads in a block can talk via shared memory and synchronize with `__syncthreads()`. Threads in *different* blocks cannot directly talk.
- **Warp** — 32 threads that execute the same instruction at the same time. NVIDIA hardware fact, not configurable. The fundamental unit of scheduling.
- **Thread** — the individual worker. Has its own registers and program counter, but always moves in lockstep with its 31 warp-mates.

**Hopper added a fifth level — Thread Block Cluster.** A group of 2–16 blocks that are guaranteed to be co-resident on the same GPC (GPU Processing Cluster). Threads in one block can read/write the shared memory of *other* blocks in the same cluster — this is called Distributed Shared Memory (DSMEM). FlashAttention-3 uses this to share K/V tiles across CTAs without re-loading from HBM. We'll come back to it in Topic 5 and 6.

## SIMT — what it really means

CPU threads each have their own program counter and can branch independently. **GPU threads in the same warp share one program counter.** All 32 threads execute the same instruction at the same clock cycle.

What if they need to do different things?

```c
if (threadIdx.x < 16) {
    // some warp threads
    a[i] = compute_one(i);
} else {
    // other warp threads
    a[i] = compute_two(i);
}
```

The hardware handles this with **predication** — both branches execute on all 32 threads, but each thread only "commits" results for the branch it should have taken. Net effect: the warp runs both branches sequentially, taking 2× the time. This is **warp divergence**, and avoiding it is the first optimization rule.

Predication is fine for short branches. Long divergent branches (one half of the warp doing 100 instructions, the other half doing 100 different ones) is where you lose 50% of your performance.

## Memory hierarchy (preview)

Each level here we expand on in Topic 5. For now, the cliff is what matters:

```
Level             Size (per-SM unless noted)   Latency       Bandwidth
────────────────────────────────────────────────────────────────────────
Registers         255 × 32-bit per thread       1 cycle        ~80 TB/s
Shared mem (SMEM) 228 KB (H100)                ~30 cycles     ~20 TB/s
L2 cache          50 MB (H100, chip-wide)      ~200 cycles    ~5 TB/s
HBM3              80 GB (H100, chip-wide)      ~500+ cycles   3.35 TB/s
HBM3e (H200/B200) 141/186 GB                                  4.9–8 TB/s
```

The bandwidth gap between SMEM and HBM is **~6× to 10×**. That ratio is the entire reason FlashAttention exists. A naive attention kernel that round-trips the N×N matrix to HBM is paying that 6× tax. FlashAttention keeps tiles in SMEM and pays it once per K/V chunk, not per element.

Mental tagline: **registers are free; SMEM is fast; HBM is the wall**. Most LLM kernels are memory-bound — they spend time waiting for HBM to deliver bytes, not on compute.

## Why this matters for performance

Almost every "why is my kernel slow?" answer is one of these four:

| Symptom | Underlying cause | Fix direction |
|---|---|---|
| Threads compute, then sit idle | Memory-bound — waiting on HBM | More tiling, fuse ops, reduce HBM traffic |
| Tensor cores idle even at high "occupancy" | Warp divergence or bank conflicts | Restructure so warps stay together |
| Achieved bandwidth far below peak | Uncoalesced memory access | Make warp threads read contiguous bytes |
| Low occupancy on every SM | Too few warps in flight to hide latency | Smaller blocks or fewer registers/thread |

You can't reason about any of these without the mental model above. Next topic puts it into code.

## Compute capability — knowing your GPU

Each NVIDIA GPU has a **compute capability** like SM75, SM80, SM90, SM100. This determines what features you can use:

| SM | GPU | Key features |
|---|---|---|
| SM75 | T4 (Turing, 2018) | Tensor cores (FP16) |
| SM80 | A100 (Ampere) | TF32, async copy, 3rd-gen tensor cores |
| SM86 | RTX 3090, A6000 | Same as SM80, smaller die |
| SM89 | RTX 4090, L4, L40 | FP8 (E4M3, E5M2) |
| SM90 | H100, H200 | TMA, WGMMA, thread block clusters, DSMEM |
| SM100 | B200, B100 | tcgen05, tensor memory (TMEM), FP4 |
| SM120 | RTX 5090 (consumer Blackwell) | Subset of SM100 |

Compile flag: `nvcc -arch=sm_90 ...`. If you compile for the wrong arch, you get either a runtime error ("no kernel image available") or you silently fall back to slow paths.

For this curriculum: a free Colab T4 (SM75) is enough for topics 1–4. Hopper-specific stuff (TMA, WGMMA) you'll read about, not run, unless you rent an H100 hour ($2/hr on RunPod) for the FlashAttention exercises.

## References (canonical 2026)

- **CUDA C++ Programming Guide 13.2** (Apr 2 2026) — sections 1–3. The official reference. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **GPU MODE lectures 1–3** — Andreas Köpf and Mark Saroufim, the modern beginner's path. https://github.com/gpu-mode/lectures
- **PMPP (Hwu/Kirk/El Hajj), 4th edition** — the textbook.
- **NVIDIA Hopper Architecture In-Depth** — Hopper-specific concepts (TMA, clusters, WGMMA). https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- **NVIDIA Hopper Tuning Guide 13.2** (Mar 5 2026) — short, dense, official. https://docs.nvidia.com/cuda/hopper-tuning-guide/
- **NVIDIA Blackwell Tuning Guide 13.2** — for SM100 / TMEM / tcgen05. https://docs.nvidia.com/cuda/blackwell-tuning-guide/
- **Mike Giles' Oxford CUDA course** — concise lecture notes. https://people.maths.ox.ac.uk/~gilesm/cuda/

## Pitfalls

1. **Treating warps as a configuration choice.** They are not. 32 threads = 1 warp. Always. Set block sizes that are multiples of 32.
2. **Thinking `__syncthreads()` synchronizes the whole grid.** It only syncs threads *within one block*. Cross-block sync requires either ending the kernel or, on Hopper+, cluster-level barriers.
3. **Mixing up "threads per block" and "threads in flight."** Threads-per-block is your launch config (e.g., 256). Threads in flight is what's actually resident on each SM, which depends on register usage, SMEM usage, and the SM's capacity. The latter is what "occupancy" means.
4. **Believing higher occupancy is always better.** Up to a point. Past it, more warps means less registers per warp, which can spill to local memory (slow). The sweet spot is workload-dependent.
