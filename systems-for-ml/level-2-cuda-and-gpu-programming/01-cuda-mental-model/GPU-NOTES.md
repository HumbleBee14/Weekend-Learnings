# GPU notes for this topic

A pocket reference for the GPU concepts that come up in Topic 1. Read alongside `CONCEPTS.md`.

## SM (Streaming Multiprocessor)

The "core" of a GPU, except a single SM has hundreds of compute units inside. Think of it as a mini-processor with:
- Its own register file (huge — 256 KB on H100)
- Its own shared memory / L1 cache (228 KB on H100)
- Its own warp schedulers (4 per SM on Hopper)
- Its own tensor cores (4 per SM on Hopper, doing matmul)
- Its own CUDA cores (FP32/INT32 ALUs — 128 per SM on Hopper)

A GPU has many SMs. H100 has 132. B200 has 148 (across two dies). The SM is what schedules and runs blocks.

Loose CPU analogy: an SM is like one CPU core that runs hyperthreads — except instead of 2 hyperthreads it runs up to 64 warps simultaneously, switching between them every cycle to hide latency.

## Warp

32 threads that execute the same instruction at the same time. The hardware doesn't run threads independently; it runs *warps*. Threads are an abstraction for the programmer.

Why 32? It's a hardware design choice that balances tensor core width, register file ports, and instruction issue rate. It's been 32 since Tesla (2006). Don't expect it to change.

## SIMT vs SIMD

Both stand for "one instruction, many data." The difference:

- **SIMD** (CPU's AVX, NEON): the programmer writes the vector — `_mm256_add_ps(a, b)`. You see and manage the lanes.
- **SIMT** (GPU): the programmer writes scalar code per thread; the hardware groups 32 threads into a warp and runs them together. You don't see the lanes; you see "threads."

SIMT is friendlier to write but the same constraints apply underneath.

## Tensor Cores

Specialized matmul units. Inside each SM, separate from the regular CUDA cores. They take small matrices (e.g., 16×8×16 for FP16) and compute `D = A·B + C` as one instruction.

This is why GPU matmul is so much faster than GPU "general compute" — the matmul has dedicated silicon.

Generations:
- 1st gen (Volta, V100, 2017): FP16
- 3rd gen (Ampere, A100, 2020): TF32, BF16, FP16, INT8 — "MMA" instruction
- 4th gen (Hopper, H100, 2022): FP8, larger tile sizes — "WGMMA" instruction (warp-group MMA)
- 5th gen (Blackwell, B200, 2024): FP4, FP6, larger TMEM scratchpad — "tcgen05" instruction family

In 2026, anything but a tensor-core matmul is leaving 90% of perf on the table. cuBLAS, CUTLASS, and FlashAttention all dispatch to tensor cores.

## CUDA Cores vs Tensor Cores

CUDA cores: one FP32/INT32 op per cycle each. There are many of them per SM (128 on Hopper).
Tensor cores: one matrix multiply per cycle each. There are few (4 per SM on Hopper) but each one does an enormous amount of work.

Most LLM serving traffic goes through tensor cores via the matmul ops in transformers. CUDA cores handle softmax, normalization, sampling — the "non-matmul" parts.

## HBM (High Bandwidth Memory)

The big pool of memory on the GPU package. HBM is fundamentally different from regular DRAM (DDR5 in your PC):

| | DDR5 (PC RAM) | HBM3 (H100) |
|---|---|---|
| Bus width | 64 bits | 5120 bits |
| Stacks | 1 | 5 stacks of dies |
| Per-channel bandwidth | ~50 GB/s | ~700 GB/s per stack |
| Total bandwidth | 50 GB/s | 3.35 TB/s |
| Capacity | 32–64 GB | 80 GB |
| Energy per byte | ~10 pJ | ~3 pJ |

HBM sits *on the same package* as the GPU die, connected by a silicon interposer. Wider bus, shorter wires, lower latency. It's the reason GPUs can feed thousands of threads with data.

Even with 3.35 TB/s, HBM is *still* the bottleneck for most LLM kernels. Reading model weights for a 70B FP16 model is 140 GB; at 3.35 TB/s that's 42ms minimum just to read the weights — for one decode step. This is why quantization (smaller weights) helps so much: less data to read.

## Memory hierarchy summary

```
       per thread       ┌──────────────┐
                        │  Registers   │  ~80 TB/s, ~256 per thread on H100
                        └──────────────┘
                                ↓
       per block        ┌──────────────┐
                        │  Shared mem  │  ~20 TB/s, 228 KB per SM on H100
                        │   (SMEM)     │  Threads in same block can talk via this
                        └──────────────┘
                                ↓
       chip-wide        ┌──────────────┐
                        │  L2 cache    │  ~5 TB/s, 50 MB on H100, 126 MB on B200
                        └──────────────┘
                                ↓
       chip-wide        ┌──────────────┐
                        │     HBM      │  3.35 TB/s, 80 GB on H100
                        │              │  4.9 TB/s on H200, ~8 TB/s on B200
                        └──────────────┘
                                ↓
       across GPUs      ┌──────────────┐
                        │   NVLink     │  900 GB/s/pair on H100, 1.8 TB/s on Blackwell
                        │              │  Used in multi-GPU training/inference
                        └──────────────┘
                                ↓
       across hosts     ┌──────────────┐
                        │  PCIe / IB   │  64 GB/s (PCIe 5.0 x16), 400+ Gb/s InfiniBand
                        └──────────────┘
```

Each level is roughly **5–10× slower** than the one above it. Topic 5 expands on this.

## Useful 2026 quick numbers to keep in your head

```
                        H100 SXM     H200 SXM     B200 SXM     MI300X
─────────────────────────────────────────────────────────────────────
HBM capacity            80 GB        141 GB       186 GB       192 GB
HBM bandwidth           3.35 TB/s    4.89 TB/s    8 TB/s       5.3 TB/s
SMs / CUs               132 SMs      132 SMs      148 SMs      304 CUs
SMEM per SM             228 KB       228 KB       228 KB       64 KB LDS
L2 / Infinity cache     50 MB        50 MB        126 MB       256 MB
FP16 dense (TFLOPS)     989          989          ~2200        1300
FP8 dense (TFLOPS)      1979         1979         ~4500        2600
FP4 dense (TFLOPS)      —            —            ~9000        —
TDP                     700 W        700 W        1000 W       750 W
```

(Numbers are approximate, dense not sparse, single GPU. NVLink5 on Blackwell is 1.8 TB/s/GPU.)
