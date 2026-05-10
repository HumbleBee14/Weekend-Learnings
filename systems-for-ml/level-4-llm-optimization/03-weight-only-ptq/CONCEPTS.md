# 03 — Weight-Only Post-Training Quantization

## When to use weight-only PTQ

Activations stay in BF16/FP16. Only weights compressed to 4 bits (or 3, 8, etc.). Use when:

- Your hardware doesn't support FP8/FP4 tensor cores (Ampere, consumer cards)
- You want maximum quality at small bit widths (4-bit weights with 16-bit activations is more forgiving than W4A4)
- The kernels for your activation precision aren't mature yet

The cost: less compute speedup than W4A4. But for memory-bound decode, the memory savings alone are most of the win.

## The four methods that matter in 2026

```
              quality    speed of quantization    notes
─────────────────────────────────────────────────────────────────────
AWQ           best       fast (5-10 min)          The 2026 default at 4-bit weight-only
GPTQ          good       slower (10-30 min)       Persistent because of tooling
HQQ           good       very fast (~1 min)       Data-free; no calibration set needed
AutoRound     best       slow (30+ min)           Intel; competitive with AWQ on quality
```

**The 2026 verdict**: AWQ effectively won at 4-bit weight-only. GPTQ persists because it's been around longer and has wider tooling support. HQQ is the "I just want to compress quickly" choice. AutoRound matches AWQ but is slower to apply.

## AWQ — Activation-aware Weight Quantization

Insight: not all weights are equal. The weights connected to *high-magnitude activations* (the "salient channels") matter more for quality. Protect those by giving them larger scaling factors during quantization.

```
Standard W4 quant:   round all weights to nearest of 16 levels
AWQ:                 measure activation magnitudes per channel during calibration
                     → for high-activation channels, scale weights up before quantizing
                     → this allocates more quantization "resolution" where it matters
```

Result: 1-2% better MMLU than uniform W4 at the same bit width. Negligible runtime overhead (the scales are baked in).

Best for: chat/instruction-tuned models. The salient channels are well-defined.

## GPTQ — second-order weight quantization

Insight: when you quantize one weight, the error can be partially compensated for by adjusting the *other* weights in the same row. Treat quantization as a layer-by-layer optimization problem with a Hessian-based update.

```
For each linear layer:
  Compute the Hessian (using a calibration set)
  Quantize weights one block at a time
  After each quantization, adjust the remaining weights to compensate
```

Slightly worse than AWQ on most chat models, comparable on some. Slower to apply. The default for GPTQModel/AutoGPTQ tooling.

## HQQ — Half-Quadratic Quantization

Data-free. No calibration set. Solves a min-max problem analytically using an iterative shrinkage approach. ~50× faster than GPTQ at competitive accuracy.

When to use: you want to ship a quantized model fast and don't have a clean calibration set. Or you have hundreds of fine-tunes and don't want to calibrate each one.

## SmoothQuant — preprocessing for activation quantization

Different problem. SmoothQuant doesn't quantize anything itself — it *preprocesses* the model so that subsequent activation quantization (W8A8, INT8 weights+activations) works better.

The trick: outliers tend to cluster in specific activation channels. SmoothQuant pushes the difficulty from activations *into* weights via a per-channel scale `s`:

```
Original:    Y = X · W
SmoothQuant: Y = (X / s) · (s · W)    [mathematically identical]
```

Pick `s` so that activations get smaller (smoother distribution → easier to quantize) and weights get slightly more outlier-y (still quantizable).

In 2026: composes with GPTQ as a preprocessing step. Use SmoothQuant first, then GPTQ. Sometimes called SpinQuant when paired with weight rotations.

## llm-compressor in 2026 — the recipe library

`vllm-project/llm-compressor` is the canonical implementation. Single library. All four methods. Aligns with vLLM's expected weight format.

```python
from llmcompressor.modifiers.quantization import GPTQModifier, AWQModifier
from llmcompressor.transformers import oneshot

# AWQ (need calibration data)
recipe = AWQModifier(
    bits=4,
    group_size=128,
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)

# GPTQ (need calibration data)
recipe = GPTQModifier(
    bits=4,
    group_size=128,
    targets="Linear",
)

oneshot(model=model, recipe=recipe, dataset=calib_dataset, num_calibration_samples=512)
model.save_pretrained("./model-w4")
```

vLLM auto-detects the quantization method from the saved config.

## Group size — the lever

`group_size=128` means: every 128 consecutive weights share one scale. Bigger group = smaller storage overhead, slightly worse quality. 128 is the 2026 standard. Some recipes use 32 or 64 for higher quality.

## What `lm_head` does and why we ignore it

The output projection (vocab_size × hidden_dim, often 150k+ tokens) is sensitive to quantization. Quantizing it can hurt rare-token quality. Standard practice: leave `lm_head` in BF16/FP16. Cost is small (one extra layer in higher precision); quality win is real.

## Pitfalls

1. **Calibrating on the wrong domain.** Chat-tuned model calibrated on Wikipedia → bad quality on chat. Use UltraChat or a domain-matched dataset.
2. **Using the same calibration data as your eval data.** Inflates measured quality. Use disjoint calibration and evaluation sets.
3. **Comparing AWQ vs GPTQ at different bit widths.** Always compare at the same bits + group size.
4. **Forgetting that activation precision matters.** W4A16 != W4A4. State both.
5. **Skipping the lm_head exclusion.** Quantizing it can hurt quality, especially on small vocabs.

## Quick recipe comparison (2026 defaults)

```
Method      Recipe                                           Calibration?    Speed
─────────────────────────────────────────────────────────────────────────────────
AWQ         AWQModifier(bits=4, group_size=128)              Yes (~512 seq)   Fast
GPTQ        GPTQModifier(bits=4, group_size=128)             Yes (~512 seq)   Slow
HQQ         (via the hqq library)                             None             Very fast
SmoothQuant Preprocess + GPTQ/AWQ                             Yes              Slow
```

For 2026 production with 4-bit weights: start with AWQ. Try HQQ if you want to skip calibration. Drop to GPTQ only if AWQ tooling isn't available.

## References

- `llm-compressor` — https://github.com/vllm-project/llm-compressor
- AWQ paper — https://arxiv.org/abs/2306.00978
- GPTQ paper — https://arxiv.org/abs/2210.17323
- HQQ — https://mobiusml.github.io/hqq_blog/
- SmoothQuant — https://arxiv.org/abs/2211.10438
- AutoRound (Intel) — https://github.com/intel/auto-round
- HQQ vs AWQ vs GPTQ comparison — https://kaitchup.substack.com/p/a-comparison-of-5-quantization-methods
