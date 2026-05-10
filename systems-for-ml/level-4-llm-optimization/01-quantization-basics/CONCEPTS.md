# 01 — Quantization Basics

## What quantization is

Reducing the number of bits used to store weights (and sometimes activations). The model gets smaller. Memory traffic gets lower. On hardware with native low-precision tensor cores, compute also gets faster.

The catch: lower precision can hurt quality. The whole field is about pushing precision down without breaking the model.

## The bit-width hierarchy in 2026

```
FP32 (32-bit)   ← baseline, almost never used in production for inference
BF16 (16-bit)   ← training default since 2022, common for high-quality inference
FP16 (16-bit)   ← legacy; same size as BF16, narrower exponent range. Edge/consumer regime.
FP8  (8-bit)    ← Hopper+ inference default in 2026 (E4M3 forward, E5M2 gradients)
INT8 (8-bit)    ← weight-only fallback when FP8 hardware unavailable; legacy lineage
NVFP4 (4-bit)   ← Blackwell native, two-level scaling
MXFP4 (4-bit)   ← OCP standard, simpler scaling. Multi-vendor.
INT4            ← weight-only via AWQ/GPTQ; activations stay higher
2/3-bit         ← extreme. Either GGUF i-quants (deployable) or BitNet 1.58 (research)
```

**Important shift since 2024**: FP8 and FP4 are not "experimental" anymore. They're production. BF16 is becoming the *high-precision* setting; FP8 is the default for inference on Hopper and Blackwell.

## BF16 vs FP16 — same size, different shape

Both are 16 bits. Different layouts:

```
FP16:  1 sign bit | 5 exponent bits | 10 mantissa bits
       Range: ±65504, precision ≈ 1e-3

BF16:  1 sign bit | 8 exponent bits |  7 mantissa bits
       Range: ±3.4e38 (same as FP32), precision ≈ 1e-2
```

BF16 trades precision for range. For ML workloads this is the right trade — gradients can have huge dynamic range; rounding errors in the 7th significant digit don't matter.

In 2026:

- **Training**: BF16 wherever possible. FP16 only on hardware that lacks BF16 support (some consumer GPUs, T4 era).
- **Inference**: BF16 for highest quality, but FP8 is the new production default.

## INT8 vs FP8 — the production split

```
INT8:    integer quantization. Weights are integers in [-128, 127], plus a per-tensor or per-channel scale.
         Software path: works on any GPU. Hardware accel only on Volta+.
         Quality: needs careful calibration; outliers can blow up the dynamic range.

FP8:     floating-point with 8 bits, two formats:
           E4M3: 4 exp + 3 mantissa, range ±448  → forward pass (weights, activations)
           E5M2: 5 exp + 2 mantissa, range ±57344 → backward (gradients have wider dynamic range)
         Hardware: H100, H200, B100, B200, MI300X. Not on Ampere.
         Quality: better than INT8 on activations because the floating-point exponent handles outliers.
```

The 2026 production rule: **if your GPU supports FP8, use FP8. Otherwise INT8 weight-only is the fallback.** We cover FP8 in depth in Topic 02.

## Why decode is what quantization helps most

From Level 3's roofline: decode is memory-bound. Each step reads the entire model from HBM to compute one new token. Halve the model size (BF16 → FP8) and decode roughly doubles in throughput on memory-bound regimes. Quarter it (BF16 → FP4) and decode ~4×.

Prefill is compute-bound. Quantization helps less there — until the compute itself can run faster (FP8 tensor cores are 2× FP16; FP4 is 2× FP8 on Blackwell).

This is why quantization is *the* lever for serving cost. Same model, quartered cost-per-token.

## The three things that can be quantized

Be precise about what's getting quantized. The vocabulary:

- **Weight-only quantization** — weights compressed to N bits; activations stay higher (FP16 or BF16). Simple, common, no hardware support needed beyond some matmul kernels.
- **Activation quantization** — activations *also* compressed. Bigger memory savings but harder; activations have outlier values.
- **KV cache quantization** — only the KV cache (cached attention keys/values) compressed. Different from weights/activations because it grows with sequence length.

A "FP8" model usually means weights+activations in FP8 (W8A8). "FP4" usually means weights in FP4 with activations in BF16 (W4A16) — a hybrid we cover in Topic 02.

## The quality-vs-size frontier

Approximate trade-offs for 2026 LLMs:

```
Format       Bits per weight   Quality vs BF16 baseline   Hardware
─────────────────────────────────────────────────────────────────
BF16         16                100% (baseline)            Anywhere
FP8          8                 99%+                       Hopper/Blackwell
INT8 (W8A16) 8                 98-99%                     Anywhere
NVFP4        ~4.25 (with scales) 95-98%                   Blackwell
MXFP4        ~4.25             93-97%                     Blackwell, MI355
INT4 AWQ     4                 95-97%                     Anywhere
INT3/IQ3_M   ~3                90-94%                     Anywhere (slower)
INT2/IQ2_M   ~2                70-85%                     Anywhere (much slower, may break)
BitNet 1.58  ~1.6              Research; ~98% per paper   Specialized kernels only
```

## What you'll actually do in this topic

Three measurements:

1. **Establish the BF16 baseline** — run a small model (Qwen2.5-0.5B or 1.5B), measure: throughput (tok/s), memory used, MMLU score.
2. **Run the same model in FP16, INT8 weight-only.** Compare same three numbers.
3. **Plot.** This is the start of the table you'll fill in across Topics 02-05.

You're not implementing quantization here — you're using existing recipes (`bitsandbytes`, `llm-compressor`) and measuring the trade-offs. Implementation lives in Topics 02-04.

## Pitfalls

1. **"Feels fine" testing.** Chat outputs look good in 5 examples; quantized model is broken on long-context or rare-token paths. Always run lm-eval-harness or KL-divergence (Topic 06).
2. **Comparing INT8 to BF16 numbers without specifying what's quantized.** "INT8" can mean W8A8, W8A16, or weight-only-with-FP8-activations. State both.
3. **Ignoring the calibration set.** PTQ methods (AWQ, GPTQ — Topic 03) calibrate on a small dataset. Wrong calibration data = bad quantization. Use the same domain you'll deploy on.
4. **Trusting per-channel int8 to "just work" on activations.** Activation quantization is hard because of outliers (one big value blows the scale). FP8's exponent handles this; INT8 needs SmoothQuant or careful clipping.
5. **Forgetting that quantized formats need quantized kernels.** A model quantized to NVFP4 needs NVFP4 GEMM kernels. Without them, you'll dequantize-on-the-fly and lose all the speed.

## References

- NVIDIA Transformer Engine FP8 primer — https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- HuggingFace quantization guide — https://huggingface.co/docs/transformers/quantization
- bitsandbytes (LLM.int8, NF4 baseline) — https://github.com/bitsandbytes-foundation/bitsandbytes
- llm-compressor (vLLM-aligned recipes) — https://github.com/vllm-project/llm-compressor
