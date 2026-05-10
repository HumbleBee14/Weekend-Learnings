# 08 — Kernel Fusion

## Files

- `CONCEPTS.md` — why fusion eliminates HBM round-trips, what can/can't fuse, the 2026 toolbox (torch.compile + Liger-Kernel + FlashInfer + custom CUTLASS)
- `swap_in_liger.py` — measure stock PyTorch vs Liger-Kernel vs torch.compile on Qwen2.5

## Quickstart

```bash
pip install torch transformers liger-kernel
python swap_in_liger.py
```

## Expected output

```
config                                       time      tok/s    peak mem
--------------------------------------------------------------------------------
stock PyTorch layers                         2150ms     59.5     3120 MB
Liger-Kernel (RMSNorm + RoPE + SwiGLU)       1730ms     74.0     2980 MB
torch.compile (reduce-overhead)              1480ms     86.5     3120 MB
```

Numbers vary. The pattern: Liger-Kernel beats stock by 20-30% on memory-bound decode. torch.compile is competitive or better for some patterns; the two are *additive* — apply Liger first, then torch.compile.

## What you should walk away with

- Fusion's win is HBM round-trip elimination, not compute speedup
- 2026 production inference uses Liger-Kernel + torch.compile + FlashInfer in combination
- Don't write your own fused kernel until you've profiled and verified the existing tools aren't enough

## Try

- **Compose Liger + torch.compile.** Apply `apply_liger_kernel_to_qwen2` first, then `torch.compile(model)`. Measure stack effect.
- **Profile it** (Level 3 tools). With Liger applied, you should see `liger_*` kernel names dominate the RMSNorm/SwiGLU paths.
- **Try a bigger model** (Qwen2.5-7B). The fusion win grows with model size because HBM pressure is higher.

## Where this goes

Topics 09-12 are the KV cache sub-arc. Build a paged KV cache yourself; understand why vLLM's design wins. The combination of fused kernels (this topic) + paged KV (next topics) + continuous batching (Topic 14) + quantization (Topics 02-05) is what makes `mini-vllm` competitive with the real thing.
