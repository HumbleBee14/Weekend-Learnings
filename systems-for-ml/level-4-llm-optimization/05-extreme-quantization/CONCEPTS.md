# 05 — Extreme Quantization

## What "extreme" means here

Below 4 bits. The territory where quality degrades visibly and engineering choices matter a lot. This topic frames extreme quantization honestly: what's deployable, what's research, where the field is in 2026.

Three regimes:

```
Bits per weight    What it is                    Status in 2026
─────────────────────────────────────────────────────────────────
3-bit              IQ3_M, IQ3_S, IQ3_XXS         Deployable on big models (≥30B)
2-bit              IQ2_M, IQ2_XS, IQ2_XXS        Deployable for 70B+, edge of usability
1.58-bit           BitNet b1.58                  Research only; specialized kernels
1-bit              IQ1_S                         Niche; quality often unusable
```

## 3-bit — usable for big models

Below 4 bits, i-quants (Topic 04) win clearly over K-quants. The importance-matrix-based allocation is what makes 3 bits work.

When 3-bit is reasonable:

- Model is ≥30B. The bit-precision-vs-parameter-count trade-off generally favors more parameters at lower bits when the model is big enough.
- You're memory-constrained and can't fit 4-bit. Common on consumer hardware running 70B+ models.

For 7B models, 3-bit is usually worse than running a smaller model at higher precision.

## 2-bit — the 70B-on-laptop trick

`IQ2_M` and `IQ2_XS` are how a 70B model fits on a 32GB Mac. ~17-20 GB instead of ~140 GB at BF16.

Quality drops noticeably — easily 5-10 points on MMLU vs BF16. But the alternative is "no model" or "smaller model at higher precision," and a quantized 70B often beats a 7B at full precision on hard tasks.

Trade-off worth the cost only when:

- Quality on hard reasoning matters (a quantized 70B's ceiling is higher than a 7B's)
- Memory is the bottleneck
- Speed matters less than quality

## BitNet b1.58 — research, not production

The 1.58-bit hypothesis: weights take only values from `{-1, 0, +1}`. log2(3) ≈ 1.58 bits per weight.

The bet: if you *train from scratch* with these constraints, the model adapts. Modern BitNet papers show 1.58-bit BitNet matching FP16 baselines at the same parameter count.

**Status in May 2026**:

- Microsoft released `BitNet b1.58 2B4T` (a 2B model trained on 4T tokens). Within 1-2 points of full-precision peers on MMLU/GSM8K.
- `bitnet.cpp` provides specialized inference kernels. **x86_64 with AVX2/AVX512 only.** ARM64 has compiler bugs as of 2026.
- HuggingFace Transformers loading gives you NO speed/energy benefit. You must use bitnet.cpp.
- **No major lab has released a frontier (>>10B) BitNet model.**

What this means for the curriculum: read the paper, run the model if you're curious, but **don't deploy it**. The hypothesis works at small scale. Frontier-scale validation is missing. Microsoft explicitly does not recommend production use.

## Why BitNet matters anyway

Even if you don't deploy: the trajectory is interesting. If 1.58-bit training validates at 70B+ scale, the energy-and-cost math for inference changes dramatically. Specialized hardware (binary/ternary tensor cores) becomes worth building. Watch the field.

## What you'd actually run for "minimum viable inference"

Practical 2026 advice for fitting big models on small hardware:

```
Hardware                              Model                Quantization
─────────────────────────────────────────────────────────────────────────
MacBook Air M3 16GB                   8B                   IQ4_XS or Q4_K_M
MacBook Pro M3 Max 64GB               70B                  IQ4_XS, Q3_K_M, IQ3_M
Desktop with 24GB GPU (RTX 4090)      70B                  AWQ Q4 or GGUF Q4_K_M (CPU offload)
Mac Studio M2 Ultra 192GB             405B                 IQ2_M
```

The IQ2_M slot on Mac Studio is the "Llama 3.1 405B on a workstation" trick. Quality is degraded but the model runs.

## Pitfalls

1. **Believing benchmark numbers from authors.** BitNet's papers show clean comparisons. Real-world quality on long-context, agentic tasks, edge cases — much less measured.
2. **Using 2-bit for 7B models.** Almost always worse than running a smaller model at 4-bit.
3. **Skipping the imatrix on i-quants.** Especially at 2-3 bits. Bad imatrix = unusable model.
4. **Comparing tok/s across formats without normalizing for hardware features.** BitNet on bitnet.cpp is fast on AVX-512 CPUs and slow on ARM. Q4_K_M is consistent across hardware.
5. **Treating extreme quant as "free."** It's a quality trade. Always measure.

## What you should walk away with

- A clear sense of when to reach for 3-bit / 2-bit / 1.58-bit
- The right framing: 3-bit and 2-bit are deployable; BitNet is research
- The mental model: extreme quant matters most for big models on small hardware
- The understanding that nothing replaces measuring quality (Topic 06)

## References

- BitNet b1.58 paper — https://arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T model — https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- bitnet.cpp — https://github.com/microsoft/BitNet
- IQ2_M / IQ3_M discussion (llama.cpp PRs, r/LocalLLaMA threads) — community knowledge, no canonical writeup
- GGUF Quantization Guide 2026 — https://www.decodesfuture.com/articles/llama-cpp-gguf-quantization-guide-2026 (covers extreme quants)
