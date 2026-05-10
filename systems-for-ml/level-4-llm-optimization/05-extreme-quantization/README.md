# 05 — Extreme Quantization

## Files

- `CONCEPTS.md` — 3-bit / 2-bit / 1.58-bit territory; what's deployable vs research; when extreme quant is worth it; honest framing of BitNet b1.58

## What you do this topic

Reading-heavy. The point is to understand the field's current state, not to deploy 2-bit models in production (mostly).

Optional experiments:

```bash
# 1. Try IQ2_M on a big model that wouldn't otherwise fit
huggingface-cli download bartowski/Qwen2.5-72B-Instruct-GGUF \
    --include "*IQ2_M.gguf" --local-dir gguf-tests
# Measure: does it produce coherent output? Does quality match the file's reputation?

# 2. Run BitNet b1.58 (read-only — don't deploy)
git clone https://github.com/microsoft/BitNet
cd BitNet && pip install -r requirements.txt
# Follow their setup; download bitnet_b1_58-large checkpoint
# Compare quality + speed to a similarly-sized FP16 baseline
```

## What you should observe

- IQ2_M on a 70B model: usable, noticeably worse than IQ4_XS on the same model. Clear ceiling above any 7B at full precision on hard tasks.
- BitNet b1.58 2B: surprisingly coherent given the constraint. Slow without bitnet.cpp's specialized kernels. Energy efficiency story is real but only realized through those kernels.

## The 2026 honest framing

Don't deploy BitNet. Do read the paper. Do follow the field — if 1.58-bit validates at 70B+ scale, the inference economics shift dramatically.

For deployable extreme quant in 2026:
- **3-bit**: IQ3_M is the practical floor for everyday use on big models
- **2-bit**: IQ2_M for "I need 70B+ to fit on this hardware"
- **Below 2 bits**: research only

## Where this goes

Topic 06 — measuring quality. After this whole quantization sub-arc, Topic 06 is what makes it all rigorous.
