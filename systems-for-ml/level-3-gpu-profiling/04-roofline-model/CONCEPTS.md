# 04 — Roofline Model

## What it is

A single chart that tells you whether a kernel is compute-bound, memory-bound, or latency-bound — and how close it is to the achievable ceiling.

Williams, Waterman, Patterson 2009. Still the load-bearing mental model in 2026.

## The chart

```
Performance
(TFLOPS)
    │
    │  ─────────────────────  ← peak compute (the flat ceiling)
    │            ╱
    │           ╱  ← compute-bound regime
    │          ╱
    │         ╱
    │        ╱
    │       ╱  ← slope = peak HBM bandwidth (the slanted ceiling)
    │      ╱
    │     ╱  ← memory-bound regime
    │    ╱
    │   ╱
    │  ╱
    │ ╱
    │╱
    └────────────────────────  →  Arithmetic intensity (FLOPs per byte)
       low                     high
```

Two ceilings: a slanted one (memory bandwidth) and a flat one (compute peak). They meet at a "ridge point" — the arithmetic intensity at which the kernel transitions from memory-bound to compute-bound.

## Two numbers per kernel

For any kernel, you compute two things:

1. **Arithmetic Intensity (AI)** = total FLOPs / total bytes read+written from HBM
2. **Achieved Performance** = total FLOPs / kernel runtime, in TFLOPS

Plot the (AI, Performance) point. Where it sits relative to the roofline tells you everything:

- **On or near the slanted line** → memory-bound, well-tuned. Win = reduce bytes (fusion, quantization).
- **On or near the flat line** → compute-bound, well-tuned. Win = lower precision (FP16→FP8), better algorithm.
- **Far below either ceiling** → latency-bound or just badly written. Win = restructure the kernel.

## Computing arithmetic intensity for common LLM kernels

You should be able to estimate AI on the back of a napkin for the kernels in your stack.

### GEMM (matmul)

```
M × K matrix × K × N matrix → M × N matrix
FLOPs = 2 · M · N · K
Bytes (FP16) = 2 · (M·K + K·N + M·N)   [read both inputs once, write output once]

For M = N = K = 4096, FP16:
  FLOPs = 2 · 4096³ ≈ 137 GFLOPs
  Bytes = 2 · 3 · 4096² ≈ 100 MB
  AI ≈ 1370 FLOPs/byte   → very compute-bound on H100 (ridge ≈ 290)
```

For decode where M=1 (one new token):
```
  FLOPs = 2 · 1 · N · K = 2NK
  Bytes ≈ 2 · K · N      [reads the whole weight matrix]
  AI ≈ 1 FLOP/byte       → severely memory-bound. Weights barely reused.
```

This is why **decode is bandwidth-bound** and prefill is compute-bound. Same operation, different dimensions, different regime.

### FlashAttention forward

```
Per attention block, B batch, N seq, H heads, D head_dim:
  FLOPs ≈ 4 · B · H · N² · D     [QK^T and softmax(·)V are both N×N×D shaped matmuls]
  Bytes ≈ 2 · 4 · B · H · N · D  [read Q, K, V + write O — NOT the N×N matrix; FA tiles it]

  AI ≈ N / 2

For N = 8192:
  AI ≈ 4000 FLOPs/byte → compute-bound on long contexts
For N = 1024:
  AI ≈ 500 → still compute-bound on H100
```

This is why FlashAttention scales — it bumps AI from ~10 (naive attention with N² intermediate) to N/2.

### RMSNorm

```
N elements, fp16:
  FLOPs ≈ 3N         [sum of squares + normalize + scale]
  Bytes ≈ 2 · 2N     [read x once, read weight, write x — fp16]

  AI ≈ 0.75 FLOP/byte    → memory-bound. Always. No way around it.
```

This is why every RMSNorm in production is fused with the operation before or after it (Liger-Kernel, FlashAttention). Standalone RMSNorm wastes bandwidth.

### Softmax

```
N elements:
  FLOPs ≈ 5N    [exp, max, sum, divide]
  Bytes ≈ 4N    [read once, write once, in fp16 = 4 bytes/element total]
  AI ≈ 1.25     → memory-bound
```

Same story. Always fuse.

## 2026 hardware peaks (what to put as your ceilings)

| GPU | Peak BF16 TFLOPS | HBM bandwidth | Ridge AI (FLOP/byte) |
|---|---|---|---|
| A100 80GB | 312 | 1.94 TB/s | 161 |
| H100 SXM | 989 | 3.35 TB/s | 295 |
| H200 | 989 | 4.89 TB/s | 202 |
| B200 SXM | 2250 (BF16) / 4500 (FP8) | 8 TB/s | 281 (BF16), 562 (FP8) |
| MI300X | 1300 | 5.3 TB/s | 245 |

The ridge point is the AI at which compute-bound and memory-bound meet. **Below the ridge → memory-bound. Above → compute-bound.** Quick check: a kernel with AI = 50 on H100 is memory-bound (50 < 295). Same kernel on a GPU with weaker bandwidth might be compute-bound.

## How to compute roofline numbers from `ncu`

`ncu --set full` (or the GUI's "GPU Speed of Light Roofline Chart") plots your kernel automatically. It also reports:

```
Compute throughput        : 67.3 TFLOPS  (fp16)
Memory throughput         : 980 GB/s
Achieved arithmetic intensity : 68.7 FLOPs/byte
% of peak compute         : 6.8%
% of peak HBM             : 50.6%
```

Read it: AI = 68.7 (memory-bound regime on H100, ridge=295). Memory utilization 50% — there's headroom on bandwidth. Suggests: bigger tile size, less re-reading, or a fusion opportunity.

## Tools that auto-generate roofline

- **Nsight Compute** — `--set full` → "GPU Speed of Light Roofline Chart" section in the GUI
- **Empirical Roofline Toolkit (ERT)** from LBL — measures *actual* achievable peaks on your specific GPU (usually 70-85% of marketed peak). Run once per machine to get realistic ceilings.
- **NVIDIA HPC SDK** — `ncu --target-processes=all --section=SpeedOfLight_RooflineChart`

## The 2026 multi-tier extension

Classic roofline = one slanted ceiling (HBM). Modern hierarchical roofline plots multiple ceilings:

```
   ────────────  peak compute
      ╱
     ╱  L2 BW ceiling      ← much higher than HBM
    ╱
   ╱
  ╱
 ╱  HBM BW ceiling
╱
```

A kernel that fits in L2 (small working set, high reuse) sits high on the chart and looks compute-bound. The same kernel with a working set bigger than L2 falls onto the HBM ceiling and is memory-bound.

This explains why kernel performance depends on input size — you can cross from "L2-resident" to "HBM-resident" at some size threshold.

## The fourth regime — communication-bound

For multi-GPU work, there's a fourth regime not in the classic roofline:

```
                     ───────  peak compute
                    ╱
                   ╱
                  ╱
                 ╱  HBM BW
                ╱
               ╱  NVLink BW    ← multi-GPU within node
              ╱
             ╱  IB BW          ← cross-node
```

For TP/PP/FSDP at scale, the bottleneck is often NCCL allreduce on NVLink or IB. Same framework, different ceiling.

## Horace He's three regimes — the canonical 2026 framing

[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) reframes roofline for ML practitioners:

1. **Compute-bound**: GPU is doing useful math. Win with better algorithms or lower precision.
2. **Memory-bound**: GPU is waiting for data. Win with fusion, tiling, smaller working set.
3. **Overhead-bound**: Python/CPU can't keep the GPU fed. Win with `torch.compile`, CUDA graphs.

This is the everyday vocabulary. When someone says "the kernel is overhead-bound," they mean the GPU is idle waiting for the CPU to launch the next kernel — the "kernel launch density" pattern from Topic 01.

## Pitfalls

1. **Computing AI from theoretical FLOPs but measured bytes.** Both should be measured (from `ncu`) or both estimated. Mixing introduces errors.
2. **Using marketed peaks as the ceiling.** Real ceilings are 70-85% of marketed. ERT gives you the truth.
3. **Forgetting that AI depends on dtype.** Same matmul has 4× lower bytes in FP8 vs FP32 — its AI is 4× higher. The kernel's regime can flip with quantization.
4. **Ignoring the L2 effect.** A "memory-bound" kernel that fits in L2 is actually L2-bound, not HBM-bound. Different ceiling, different optimization.
5. **Looking at one kernel.** A workload's bottleneck is the *sum* of its kernels. A kernel that's 5% of total time can be wildly inefficient and still not matter.

## What you should be able to do

After this topic:

- Estimate the AI of any LLM kernel (GEMM, attention, RMSNorm, etc.) on a napkin
- Read `ncu`'s roofline plot and identify the regime
- Predict whether a quantization change will move the kernel into a different regime
- Use Horace He's three-regime vocabulary fluently

## References

- Williams, Waterman, Patterson 2009 (the original) — https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
- Horace He — Making Deep Learning Go Brrrr From First Principles — https://horace.io/brrr_intro.html
- Modal GPU Glossary — Roofline Model — https://modal.com/gpu-glossary/perf/roofline-model
- Nsight Compute Roofline section — https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline
- Empirical Roofline Toolkit — https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/software/ert/
