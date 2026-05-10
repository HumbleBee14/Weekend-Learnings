# 03 — MLX vs llama.cpp Metal vs PyTorch MPS

The three frameworks that can run an LLM on Apple Silicon. They are not interchangeable. Each has a substrate, a model format, a typical use case, and a 2026 verdict.

## The substrate-level differences

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                  │ MLX              │ llama.cpp Metal  │ PyTorch MPS      │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Built for        │ Apple Silicon    │ Cross-platform   │ Ports CUDA model │
│                  │ first-class      │ generic          │ to Metal         │
│ Model format     │ MLX-native /     │ GGUF only        │ PyTorch /        │
│                  │ safetensors      │                  │ safetensors      │
│ Lazy / eager     │ Lazy + fusion    │ Eager (C++)      │ Eager; compile   │
│                  │                  │                  │ partial on MPS   │
│ Quantization     │ 2/3/4/6/8-bit    │ K-quants, i-quants│ INT8 limited    │
│                  │ MLX-native, FP4  │ Unsloth Dynamic, │                  │
│                  │ on M5            │ FP4 (May 2026)   │                  │
│ KV cache quant   │ 4-bit, 8-bit     │ 4-bit, 8-bit     │ Not standard     │
│ Spec decoding    │ EAGLE-3,         │ Built-in draft   │ Manual           │
│                  │ QuantSpec        │ model param      │                  │
│ Training         │ Yes (LoRA, SFT,  │ No, inference    │ Yes, slow on LLM │
│                  │ DPO, GRPO)       │ only             │                  │
│ Day-1 model      │ days-weeks       │ hours via GGUF   │ varies           │
│ support          │ (mlx-community)  │ converters       │                  │
│ Multi-platform   │ Apple only       │ Mac, Linux, Win, │ Mac (CUDA path   │
│                  │                  │ phone, web       │ on Linux/Win)    │
│ Best at          │ Mac LLM, dev,    │ Cross-platform,  │ Non-LLM, ports   │
│                  │ research, train  │ universal client │ from PyTorch     │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

## Throughput on a 7B 4-bit model

Numbers from a 2026 M3 Max 64GB, batch=1 generation, 1024-token decode after a 128-token prompt:

```
MLX (mlx-lm)              ~230 tok/s
llama.cpp Metal           ~150 tok/s
PyTorch MPS               ~55 tok/s
```

On M5 Max (Neural Accelerators, mlx-lm with `mx.fast.matmul`):

```
MLX                       ~310 tok/s
llama.cpp Metal           ~165 tok/s   (no NA path yet)
PyTorch MPS               ~60 tok/s
```

The MLX-vs-llama.cpp gap on Apple Silicon has stayed at 50–90% generation throughput across the M2/M3/M4/M5 generations. PyTorch MPS has not closed it.

## Why the gap

**MLX advantages.**
- Lazy graph + Metal-native compiled kernels mean fewer launches and less DRAM pressure.
- KV layout optimized for unified memory from the start.
- Quantization codepaths handcrafted in Metal Shading Language.
- M5 Neural Accelerators only addressable from MLX in 2026.

**llama.cpp advantages.**
- One binary runs on every platform Apple, NVIDIA, AMD, mobile, web.
- GGUF ecosystem is enormous and instant — the moment a model is on HuggingFace, there is a GGUF.
- Battle-tested CPU path (NEON, AVX-512+VNNI, AMX, SME2). Best CPU LLM in the world.
- Speculative decoding via `--draft-model` is simple and works.
- Smaller memory overhead at idle.

**PyTorch MPS advantages.**
- You already have PyTorch code; it runs.
- Non-LLM workloads (vision, audio, classic transformers) work fine.
- `torch.compile` is improving on MPS, but in 2026 it is still partial — many ops fall back to eager.

## Memory, not just speed

Same 7B 4-bit, same 4K context, peak resident memory:

```
MLX                       ~5.2 GB
llama.cpp Metal           ~5.6 GB
PyTorch MPS               ~7.8 GB
```

PyTorch's MPS allocator and intermediate-tensor lifetime add ~50% overhead on a small model. On a 70B model that gap is the difference between fitting on 64GB and not.

## When to pick which

- **MLX** — anything LLM-shaped on a Mac that you control. Dev loop, research, fine-tuning, local serving for yourself or a team.
- **llama.cpp Metal** — you need the same binary to run on Linux servers, Windows boxes, an Android phone. Or you need a GGUF model that has not been MLX-ported yet.
- **PyTorch MPS** — non-LLM model, or a research codebase you do not want to rewrite, or a training workload that is not yet supported on MLX.

The 2024-era claim that "llama.cpp is the local choice" survives mostly out of inertia. In 2026 on Apple Silicon, MLX is the choice. llama.cpp's reason for existing is portability and CPU fallback, both of which still matter.

## Project 4 graph G18

This topic produces G18 directly: TTFT and tokens/sec for the same 4-bit Qwen2.5-7B in MLX, llama.cpp Metal, and PyTorch MPS. Same prompts, same Mac, same temperature. The bench script (`benchmark.py`) is a starting point.

## Pitfalls when benchmarking

1. **Different quantization across frameworks.** A 4-bit GGUF (Q4_K_M) and a 4-bit MLX (group_size=64) are not the same bits. Use `lm-eval-harness` on each to confirm comparable quality before comparing speed.
2. **Cold cache.** First call mmap-faults weights in. Always warm before timing.
3. **Background load.** Spotlight reindexing, Time Machine, browser tabs all eat memory bandwidth. Quit them.
4. **Power profile.** macOS throttles under thermal pressure. Run on AC power, fan-supported case, not in a long meeting.
5. **Different prompt lengths.** TTFT scales with prompt length. Match it across runs or you are measuring different things.
6. **Sampling differences.** Greedy decoding (`temp=0`) is the only reproducible setting. Temperature non-zero produces variable token counts and length-dependent timings.

## References

- MLX vs llama.cpp benchmark study (Apple): https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- llama.cpp Metal backend: https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.m
- PyTorch MPS status: https://pytorch.org/docs/stable/notes/mps.html
- mlx-lm benchmark scripts: https://github.com/ml-explore/mlx-lm/tree/main/benchmarks
- llama-bench (canonical llama.cpp tool): https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench
