# 02 — FP8 and NVFP4

## Files

- `CONCEPTS.md` — FP8 E4M3/E5M2, NVFP4 two-level scaling, NVFP4 vs MXFP4 fault line, hardware support matrix, llm-compressor v0.9 recipes
- `quantize_to_fp8.py` — FP8_DYNAMIC recipe (data-free, no calibration needed) using llm-compressor
- `quantize_to_nvfp4.py` — NVFP4 W4A4 with calibration (UltraChat for calibration data)

## Quickstart

```bash
pip install llm-compressor vllm transformers datasets

# FP8 (Hopper+) — fast, no calibration needed
python quantize_to_fp8.py
vllm serve ./Qwen2.5-0.5B-Instruct-FP8

# NVFP4 (Blackwell only for native speed) — calibration takes ~5-15 min
python quantize_to_nvfp4.py
vllm serve ./Qwen2.5-1.5B-Instruct-NVFP4
```

## Expected outcomes

Compared to BF16 baseline from Topic 01:

| Recipe | Memory | Decode throughput | Quality (MMLU) |
|---|---|---|---|
| BF16 | 100% | 1.0× | 100% |
| FP8 (Hopper+) | ~50% | ~1.7× | 99-100% |
| NVFP4 (Blackwell) | ~25% | ~3-4× | 95-98% |

The exact numbers depend on model size, hardware, batch size. Bigger models show bigger relative wins (more memory-bound).

## Try

- **Use a bigger model** (Qwen2.5-7B). FP8's win becomes more dramatic — both because the absolute memory savings are larger and because BF16 starts to bump into HBM capacity limits.
- **Vary the calibration set for NVFP4.** Calibrate on chat data, evaluate on code → quality drops on code. Domain matters.
- **Run both serving modes side by side.** Hit the FP8 server and the BF16 server with the same Locust workload. Compare TTFT and ITL.
- **Measure quality with `lm-eval-harness`** before declaring success (Topic 06).

## What you should walk away with

- Fluency in FP8 E4M3 vs E5M2 — when each matters
- Understanding of *why* NVFP4 works (two-level scaling, FP8 scale type vs MXFP4's E8M0)
- Familiarity with `llm-compressor` recipes
- Numbers in your quality-vs-cost table for FP8 and (if Blackwell available) NVFP4

## Where this goes

- Topic 03 — weight-only quantization (AWQ/GPTQ/HQQ) — the path when you can't do W8A8 or W4A4
- Topic 04 — local quant formats (GGUF) — when CPU/Mac is the target
- Topic 06 — measuring quality (essential for any of these)
