# 01 — Unified Memory Mental Model

## The substrate

Apple Silicon (M1 through M5) has no separate VRAM. CPU and GPU share the same physical DRAM through a single memory controller. There is one pool of bytes; the CPU reads from it, the GPU reads from it, the Neural Engine reads from it. No PCIe, no `cudaMemcpy`, no pinned-host buffers, no DMA dance.

```
Discrete GPU (CUDA world)              Apple Silicon (UMA world)
─────────────────────────              ─────────────────────────
   ┌────────┐                              ┌──────────────────┐
   │  CPU   │                              │  CPU GPU ANE     │
   │  DDR5  │  ◄─── PCIe Gen5  ───►        │   shared DRAM    │
   └────────┘     64 GB/s ish              │   460–614 GB/s   │
   ┌────────┐                              └──────────────────┘
   │  GPU   │                                 one address space
   │  HBM3e │  ◄── 3-8 TB/s on-card             zero-copy
   └────────┘
```

The CUDA model is a two-tier hierarchy: high-bandwidth on-card HBM, narrow PCIe link to host. Apple's model is a one-tier hierarchy: one DRAM pool, one bandwidth budget shared by every consumer.

## What this changes for LLMs

**KV cache placement is free.** A discrete GPU running a 100K-token context has to choose: keep KV on the card (fits, eats VRAM) or offload to host RAM (cheap, but every attention step pays a PCIe round trip). On M-series there is no choice — KV is in the only DRAM that exists, and the GPU reads it at full memory bandwidth.

**No staging buffers for tokenizer output, logits, samplers.** Tokenize on CPU, hand the tensor to MLX, sample on GPU, decode the new token id on CPU — all are pointer aliases into the same buffer. This is the structural reason MLX's overhead per token is lower than PyTorch's on any platform.

**Model load is `mmap` of a `.safetensors` file straight into the address space MLX uses.** The OS pages weights in lazily; the GPU sees them as soon as they are resident. Cold-start a 70B 4-bit model in seconds, not the minute-plus a CUDA load takes after `cudaMemcpy`-from-disk pipelines.

## The catch — bandwidth is shared

```
M3 Max 16-core CPU + 40-core GPU + 16-core ANE
     all draining the same memory controller
                       │
                       ▼
            ~400 GB/s effective
```

If the CPU is busy tokenizing a long prompt while the GPU is decoding, both stall on the same DRAM channels. Production code keeps tokenization off the hot path: pre-tokenize, cache, run a dedicated tokenizer thread that competes minimally with the GPU's decode loop.

The same applies to other GPU-adjacent CPU work — JSON parsing during structured decode, copy of logits to a sampler running on CPU, vector DB lookups for RAG. Every CPU memory read during decode steals from the bandwidth the GPU needs to fetch weights and KV.

## Memory-bandwidth math for decoding

Decode is memory-bound. Per token, the GPU must read approximately:

```
weights_bytes  +  kv_cache_bytes_for_this_token_position
```

For a 7B 4-bit model with 4-bit KV at 8K context:
- Weights: ~3.8 GB
- KV at 8K: ~250 MB (dropping fast with KV quant)
- Total per token: ~4 GB

On 400 GB/s effective bandwidth that ceilings at ~100 tok/s. Real M3 Max does ~230 tok/s with MLX — better than the math because MLX fuses ops and avoids re-reading layer activations through DRAM.

For 70B 4-bit:
- Weights: ~38 GB
- Per token ceiling: 400 / 38 ≈ 10 tok/s

This back-of-envelope is the single most useful number on Apple Silicon. If you know weight bytes and effective bandwidth, you know the upper bound. MLX's 230 tok/s on 7B 4-bit is ~57% of theoretical peak — high for a real system.

## Why MoE explodes the math

A 109B-param MoE like Llama 4 Scout activates ~17B per token. Memory bandwidth per token is set by *active* params, not total. So 4-bit Scout's per-token bandwidth is ~9 GB, similar to a dense 7B 4-bit at 16-bit precision. Result: ~50 tok/s on M5 Max with a 109B-total model. Topic 09 has the full math.

## Capacity vs bandwidth

These are independent dimensions and people confuse them constantly.

- **Capacity** is which model fits at all. Set by total unified memory. M3 Max 64GB fits a 70B 4-bit (~35GB) with room for KV and OS. M5 Max 128GB fits 70B fp16 or 405B 4-bit.
- **Bandwidth** is how fast it runs. Set by the memory controller. M3 Max ~400 GB/s, M3 Ultra ~800 GB/s, M5 Max similar to M3/M4.

Apple advertises "up to 614 GB/s" on Max chips. Effective is lower because the controller serves CPU + GPU + ANE concurrently. Your decode loop sees somewhere between half and all of peak depending on what else is running.

## What this means for code

1. Allocate once, reuse. Every alloc is a real allocation (no pinned-host pool to amortize against).
2. Keep tokenization, sampling, and any CPU-side post-processing tight — every CPU read during decode steals GPU bandwidth.
3. Use `mlx.core.metal.set_memory_limit` to cap MLX so the OS still has room for paging. Without it, the OS will start compressing memory and your decode tok/s drops by 2-3x silently.
4. Watch `memory_pressure` (`memory_pressure -l warn` in Terminal). When it goes yellow, you are paging. When red, your model is being evicted to swap and tok/s collapses.

## References

- Apple — exploring LLMs with MLX on M5: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- MLX architecture overview: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- Apple silicon CPU optimization guide (memory subsystem): https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide
- M-series memory bandwidth measurements (community): https://github.com/ggerganov/llama.cpp/discussions/4167
