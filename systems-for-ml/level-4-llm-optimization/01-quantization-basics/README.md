# 01 — Quantization Basics

## Files

- `CONCEPTS.md` — bit-width hierarchy, BF16 vs FP16, INT8 vs FP8, what gets quantized (weights/activations/KV), 2026 trade-off frontier
- `baseline_measurements.py` — measures BF16, FP16, INT8 weight-only, NF4 4-bit on the same model. Throughput + peak memory.

## Quickstart

```bash
pip install torch transformers bitsandbytes accelerate
python baseline_measurements.py
```

## Expected output

```
config                       tok/s     ms/tok     peak mem
----------------------------------------------------------------
BF16 (baseline)              48.3       20.7      1024.5 MB
FP16                         48.1       20.8      1024.5 MB
INT8 (W8A16, bnb)            32.1       31.2       512.3 MB
NF4 (W4A16, bnb)             28.5       35.1       256.1 MB
```

Numbers are illustrative — depend heavily on GPU. Two takeaways:

- **INT8 halves memory; NF4 quarters it.** Memory is the prize for memory-bound decode.
- **bitsandbytes kernels are not the fastest.** INT8/NF4 here is *slower* than BF16 because bnb's kernels prioritize correctness and breadth over speed. Production stacks (Topic 02 onward) use FP8/NVFP4 with optimized kernels, getting *both* memory savings *and* speed wins.

## Try

- **Use Qwen2.5-1.5B or 7B** instead of 0.5B. Memory savings become more dramatic. Speed deltas remain similar in shape.
- **Add a longer prompt** (1000 tokens). Prefill time gets included; the regime shifts toward compute.
- **Try `BitsAndBytesConfig(bnb_4bit_quant_type="fp4")`** instead of NF4. Subtly different format; small quality difference.
- **Skip ahead to Topic 02** if your GPU supports FP8 — that's where the real production wins live.

## What to take away

This topic establishes the BF16 baseline you'll compare against for every subsequent quantization method. The pattern: change *one thing*, measure throughput + memory + quality (Topic 06). Don't chase speed without checking quality.

## Where this goes

- Topic 02 — FP8/NVFP4 (production datacenter precisions in 2026)
- Topic 03 — AWQ/GPTQ via llm-compressor (the real PTQ recipes)
- Topic 04 — GGUF for local serving (i-quants, Unsloth Dynamic v2)
- Topic 05 — extreme quant (BitNet, IQ2_M)
- Topic 06 — quality evaluation (without which all of this is meaningless)
