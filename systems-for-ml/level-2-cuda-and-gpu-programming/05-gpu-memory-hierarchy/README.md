# 05 — GPU Memory Hierarchy

## Files

- `CONCEPTS.md` — every level (registers → SMEM → DSMEM → L2 → HBM → NVLink → PCIe), bandwidth/latency numbers, what each one is for, 2026 hardware specs
- `measure_bandwidth.py` — Triton script that measures achieved HBM bandwidth on your GPU at several data sizes, plus a compute-bound comparison
- **`READING-PRODUCTION-KERNELS.md`** — guided reading exercise. Clone vLLM and Liger-Kernel, navigate three real production kernels (PagedAttention CUDA C++, RMSNorm Triton, vLLM's Triton attention backend), find every concept from Topics 1–5 in real code. **The most industry-shaped exercise in this level.**

## Quickstart

```bash
pip install triton torch
python measure_bandwidth.py
```

## Expected output (varies by GPU)

```
GPU: NVIDIA H100 80GB HBM3  (compute capability sm_90)
Total memory: 80.0 GB

Bandwidth measurements (peak achievable on a streaming pattern):

Per-call bandwidth at increasing data sizes:
  HBM streaming (1 MB):  4500 GB/s   ← fits in L2, looks faster than spec
  HBM streaming (4 MB):  4200 GB/s   ← still partly L2
  HBM streaming (16 MB): 3100 GB/s   ← spilling to HBM
  HBM streaming (64 MB): 2900 GB/s   ← real HBM bandwidth
  HBM streaming (256 MB): 2850 GB/s  ← real HBM bandwidth (~85% of 3.35 TB/s spec)

  Compute heavy (64 MB):  450 GB/s   ← compute-bound, not bandwidth-bound
```

The point: at small sizes you're hitting L2; at large sizes you're hitting real HBM. The compute-bound test should be much lower bandwidth — because it's bottlenecked by the SFUs computing transcendentals, not by memory.

## Try

- **Find your L2 cliff.** Run with sizes 1MB, 2MB, 4MB, 8MB, 16MB, 32MB. Bandwidth will be high until your data exceeds L2 capacity (50 MB on H100, 4 MB on T4), then drop to HBM bandwidth.
- **Plot it.** That cliff is the L2 → HBM transition. It's the most important boundary on your GPU.
- **Compare GPUs if you can.** T4 vs A100 vs H100 → very different L2 sizes and HBM bandwidths. The shape of the curve tells you the architecture.
- **Run `nvidia-smi` while measuring.** GPU power draw spikes during streaming — proves you're saturating something.

## What to take away

When you see a slow kernel, the first question is: which level am I bottlenecked on?

- **HBM bandwidth saturated** (streaming-like access pattern, achieved BW ≈ peak HBM): you're memory-bound. Fix is fusion or quantization to reduce data traffic.
- **HBM bandwidth low but kernel is slow** (achieved BW << peak): probably uncoalesced or low-occupancy. Check Nsight Compute's "Memory Throughput" page.
- **Both BW and FLOPS look low**: you're latency-bound (waiting on memory but not enough warps in flight to hide it). Increase occupancy or reduce dependencies.
- **FLOPS saturated (close to tensor core peak)**: compute-bound. Win is a better algorithm or lower precision (FP8, FP4).

The roofline model from Level 3 makes this rigorous. This week is the foundation.

## Where this goes

Topic 6 is FlashAttention. The whole reason it exists is the bandwidth gap measured here — keeping intermediate matrices in SMEM instead of round-tripping through HBM is the entire trick.
