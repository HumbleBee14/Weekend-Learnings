# 03 — Weight-Only PTQ

## Files

- `CONCEPTS.md` — AWQ vs GPTQ vs HQQ vs SmoothQuant; the 2026 fault lines; group_size; lm_head exclusion
- `quantize_awq.py` — apply AWQ at 4-bit weights, 16-bit activations using llm-compressor

## Quickstart

```bash
pip install llm-compressor transformers datasets vllm

python quantize_awq.py    # ~5-10 min
vllm serve ./Qwen2.5-1.5B-Instruct-AWQ-W4
```

## What you'll see

Memory drops to ~25-30% of BF16 baseline. Decode speeds up 1.5-2.5×. MMLU drops 1-2 points.

For a 70B model, this is the difference between "won't fit on one H100" and "fits comfortably." The quality cost is small enough that nearly every production LLM in 2026 ships in some quantized form.

## Try

- **GPTQ instead of AWQ** — change `AWQModifier` to `GPTQModifier`. Compare quality.
- **HQQ for instant compression** — `from hqq.engine.hf import HQQModelForCausalLM`. No calibration. Sub-minute quantization.
- **Different group sizes** (32, 64, 128). Smaller = better quality, slightly more storage.
- **Calibrate on your domain.** If you serve code, calibrate on code data.

## Connection to Topic 02

Topic 02 (FP8/NVFP4) needs Hopper+/Blackwell. This topic (W4A16) runs on any GPU. They're not competing — they're for different deployment targets.

A common 2026 production pattern: deploy FP8 on H100/H200 for the main service; deploy AWQ-W4 on consumer/edge for cheaper tiers. Both from the same base model.

## Where this goes

- Topic 04 — local/CPU formats (GGUF i-quants)
- Topic 06 — measuring quality properly so this table isn't lies
