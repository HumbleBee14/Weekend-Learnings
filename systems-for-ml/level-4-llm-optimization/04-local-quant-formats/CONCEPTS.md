# 04 — Local Quant Formats

## When this matters

Datacenter (Topics 02-03): FP8, NVFP4, AWQ on Hopper/Blackwell GPUs. Big batch, sustained load.

Local: laptops, desktops, Apple Silicon, consumer GPUs. Single user. CPU might be the only option. Apple Silicon's unified memory architecture changes the math.

This topic covers formats designed for local: **GGUF** (llama.cpp's format) and **EXL2** (ExLlamaV2). Different design constraints, different trade-offs.

## GGUF — the dominant local format in 2026

GGUF (GPT-Generated Unified Format) is llama.cpp's binary container. Single file. Memory-mappable. Quantization metadata baked in.

The quantization variants form a forest:

### K-quants (K-quantization)

- `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q4_K_S`, **`Q4_K_M`**, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`

Block-quantized with super-blocks. Each super-block has its own scale; sub-blocks share a finer scale. The S/M/L suffix is "small/medium/large" — finer quantization granularity.

**`Q4_K_M` is the 2026 default for everyday local use.** ~4.85 bits per weight, decent quality, fast on CPU.

### i-quants (importance-quantization)

- `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `IQ3_M`, **`IQ4_XS`**, `IQ4_NL`

Uses an *importance matrix* (imatrix) computed from a calibration dataset. Weights with higher importance get more bits. Result: **higher quality at the same average bit width**, especially below 4 bits.

Cost: slower decode on CPU (more complex dequantization). Depends heavily on imatrix quality.

### When K-quants vs i-quants in 2026

```
Bit width        Recommendation           Notes
─────────────────────────────────────────────────────────────────
8-bit            Q8_0                     Reference; almost no quality loss
6-bit            Q6_K                     Sweet spot for "max quality, still small"
~4.85 bit        Q4_K_M                   Default everyday choice
~4.4 bit         IQ4_XS                   ~9% smaller than Q4_K_M, slower CPU decode
~3.5 bit         IQ3_M / Q3_K_M           IQ3_M better quality
~3 bit           IQ3_XXS / IQ3_S          i-quants pull ahead clearly here
~2.5 bit         IQ2_M                    Aggressive but usable for 70B+
~2 bit           IQ2_XS / IQ2_XXS         Quality drops noticeably; usable as last resort
```

The 2026 rule: **at ≤3 bits, always use i-quants. At 4 bits, Q4_K_M is the safe default.**

CPU decode speed matters: i-quants are slower than K-quants on CPU because dequantization is more complex. On GPU (Metal, CUDA backends in llama.cpp) the gap closes.

## Unsloth Dynamic v2.0 GGUFs — the 2026 frontier

Unsloth (the team behind the fast fine-tuning library) released "Dynamic v2.0 GGUFs" in 2025-2026. They use **per-layer mixed bit widths** based on sensitivity analysis:

```
Standard Q4_K_M:    every layer at ~4.85 bpw
Unsloth Dynamic:    sensitive layers at 6 bpw, robust layers at 3 bpw, average ~4.5 bpw
```

Result: smaller average size *and* better quality than uniform K-quants or i-quants on KL-divergence benchmarks. The current state-of-the-art for local GGUF in 2026.

When to use: you're serving a model locally and quality matters. The size win plus quality win is real. Available on HuggingFace as `unsloth/<model>-GGUF`.

## FP4 in llama.cpp (new in 2026)

llama.cpp added native FP4 support — both NVFP4 and MXFP4 formats — in 2026. Local users now have access to the same quantization the datacenter uses.

This matters for: Apple Silicon M-series (especially M5 with neural accelerators), modern consumer NVIDIA cards on llama.cpp's CUDA backend, and AMD via ROCm.

## EXL2 — increasingly niche

ExLlamaV2's format. Mixed bit widths per layer (similar idea to Unsloth Dynamic, predates it). Optimized for consumer NVIDIA GPUs.

In 2026: niche. vLLM and SGLang dominance pulled production attention away from ExLlamaV2. EXL2 still has fans in r/LocalLLaMA but newer projects target GGUF.

## Practical: how to convert

```bash
# Convert a HuggingFace model to GGUF (FP16/BF16, no quantization yet)
python llama.cpp/convert_hf_to_gguf.py /path/to/Qwen2.5-1.5B-Instruct \
    --outfile model-bf16.gguf

# Quantize to Q4_K_M (the everyday default)
./llama.cpp/build/bin/llama-quantize model-bf16.gguf model-q4-k-m.gguf Q4_K_M

# Quantize to IQ4_XS (smaller, slightly worse CPU speed)
./llama.cpp/build/bin/llama-quantize model-bf16.gguf model-iq4-xs.gguf IQ4_XS
# Note: i-quants need an imatrix file for best results:
./llama.cpp/build/bin/llama-imatrix -m model-bf16.gguf -f calibration.txt
./llama.cpp/build/bin/llama-quantize --imatrix imatrix.dat model-bf16.gguf model-iq3-m.gguf IQ3_M
```

For Unsloth Dynamic GGUFs: just download from HuggingFace — `unsloth/Qwen2.5-1.5B-Instruct-GGUF` etc.

## Hardware considerations

```
Hardware                  Best format            Notes
─────────────────────────────────────────────────────────────────
Mac (M1-M5)               GGUF Q4_K_M or         Metal backend, all GGUF formats supported
                          MLX-converted          MLX faster on M-series; see Level 8
Consumer NVIDIA (RTX)     GGUF + CUDA backend    or AWQ via vLLM if you want server semantics
Linux CPU only            GGUF Q4_K_M (K-quants  i-quants slower on CPU
                          for speed) or IQ4_XS
                          (for size)
Edge (ARM phones, RPi)    GGUF Q4_0 or smaller   bitnet.cpp for BitNet 1.58 (research)
```

## Pitfalls

1. **Comparing GGUF Q4 to AWQ Q4 by file size.** Different storage layouts. Compare by quality at the same average bpw (use KL-divergence — Topic 06).
2. **Using i-quants without an imatrix.** Quality drops sharply. Always compute imatrix from a representative calibration set.
3. **Calibration set domain mismatch.** Same as Topic 03 — calibrate on the domain you'll deploy on.
4. **Forgetting to update llama.cpp regularly.** Quantization quality has improved meaningfully in 2025-2026; stale builds use old methods.
5. **Trusting filename suffixes.** `Q4_K_M` from one repo might be calibrated; from another, raw round-to-nearest. Quality varies. Prefer reputable sources (Unsloth, lmstudio-community, bartowski).

## References

- llama.cpp — https://github.com/ggml-org/llama.cpp
- GGUF Quantization Guide 2026 — https://www.decodesfuture.com/articles/llama-cpp-gguf-quantization-guide-2026
- Unsloth Dynamic 2.0 GGUFs — https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- Bartowski's quants on HF — https://huggingface.co/bartowski (community-respected source)
- ExLlamaV2 — https://github.com/turboderp-org/exllamav2
