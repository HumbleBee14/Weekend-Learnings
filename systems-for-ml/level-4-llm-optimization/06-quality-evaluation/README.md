# 06 — Quality Evaluation

## Files

- `CONCEPTS.md` — KL divergence vs perplexity, why KL won in 2026, lm-eval-harness, per-layer sensitivity, the quality-vs-cost table
- `measure_kl_divergence.py` — KL between quantized and BF16 reference, the modern quality metric
- `run_lm_eval.sh` — task-suite eval (MMLU, GSM8K, ARC, HellaSwag) for a recipe vs reference

## Quickstart

```bash
pip install torch transformers datasets lm-eval vllm

# KL divergence (fast; ~5 min for 256 samples)
python measure_kl_divergence.py --quantized ./Qwen2.5-1.5B-Instruct-FP8

# Task suite (slow; 30+ min for full MMLU)
chmod +x run_lm_eval.sh
./run_lm_eval.sh ./Qwen2.5-1.5B-Instruct-FP8
```

## Expected output

```
Mean per-token KL divergence: 0.012
```

For FP8 dynamic, KL ~ 0.01-0.02 is typical. If your KL is much higher, something's wrong with the recipe (calibration domain mismatch, bad observer, etc.).

For lm-eval, MMLU should drop <1 percentage point for a good FP8 recipe; ~1-2 pp for AWQ-W4; 2-4 pp for NVFP4.

## Try

- **Run on every recipe from Topics 02-05.** Build the full quality-vs-cost table for your model.
- **Compare your calibration domain matters.** Calibrate AWQ on UltraChat, eval on code (HumanEval). Quality drop is visible. Re-calibrate on code data, re-eval — gap closes.
- **Per-layer sensitivity sweep.** For each layer, measure KL with only-that-layer-quantized. Identify the most sensitive layers. This is the basis for mixed-precision recipes.
- **MMLU-Pro (harder)** instead of MMLU. Quantization differences become more visible on harder tasks.

## What this topic enforces

The quantization sub-arc (Topics 01-05) produced numbers like "FP8 is 1.7× faster." Without this topic, those numbers are PR. With this topic — KL + lm-eval — they're engineering.

The 2026 production rule: **no quantization recipe ships without KL + at least one task-suite eval.** This applies whether you're a frontier lab or a startup.

## Where this goes

The quantization sub-arc closes here. From Topic 07 onward we're back to the kernel/serving/cache layer:

- Topic 07 — torch.compile (Dynamo + Inductor + piecewise CUDA graphs)
- Topic 08 — kernel fusion principles
- Topics 09-12 — KV cache (the heart of `mini-vllm`)
- Topics 13, 17 — speculative decoding
- Topic 14 — continuous batching
- Topic 15 — structured output
- Topic 16 — serving concurrency primitives
