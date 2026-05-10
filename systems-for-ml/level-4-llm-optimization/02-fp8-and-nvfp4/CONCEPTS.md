# 02 — FP8 and NVFP4

## What changed since 2024

- **FP8 is the 2026 datacenter inference default** on Hopper and Blackwell.
- **NVFP4 (Blackwell native FP4) shipped at scale** in 2025-2026. It's no longer experimental.
- **MXFP4 (the OCP standard) is now production-viable** thanks to MR-GPTQ (ICLR 2026) solving the calibration quality gap.
- **MXFP6 exists** but rarely used — teams jump from MXFP8 to NVFP4.

The two-level scaling in NVFP4 is the most important new concept.

## FP8 — the two formats

```
E4M3:  1 sign | 4 exponent | 3 mantissa     range ±448, finer precision
E5M2:  1 sign | 5 exponent | 2 mantissa     range ±57344, coarser precision
```

The 2026 production split is settled:

- **E4M3 for forward** (weights, activations) — finer precision matters more here
- **E5M2 for backward** (gradients) — wider range matters because gradients can blow up

NVIDIA's Transformer Engine ships this as `HYBRID` mode. Inference-only deployments use **E4M3 everywhere** — there's no backward pass.

For an inference engineer in 2026: when someone says "FP8" without qualifying, they mean E4M3.

## How FP8 quantization works mechanically

```
Original FP16/BF16 tensor → divide by per-tensor scale → round to nearest E4M3 representable
                                                          ↓
                                           Store: int8-style FP8 values + the FP32 scale
                                                          ↓
At GEMM time: tensor cores execute FP8 × FP8 → FP32 accumulator → multiply by scale → output
```

The scale is **per-tensor** (one scale for the whole weight matrix) or **per-row** (one scale per row, finer-grained). Per-row is more accurate, slightly more expensive.

Calibration finds the scale: `scale = max(|x|) / E4M3_max`. Static calibration uses a fixed dataset; dynamic computes scales per-batch (rare in inference because it adds overhead).

## NVFP4 — two-level scaling is the breakthrough

Naive FP4 has 16 representable values total (E2M1: 1 sign + 2 exp + 1 mantissa). Range is tiny. A single outlier ruins the whole tensor.

NVFP4 fixes this with **two-level scaling**:

```
Level 1 (per-block):     16-element block has its own FP8 E4M3 scale
                         (not E8M0 like MXFP4 — this is the key difference)

Level 2 (per-tensor):    one FP32 global scale on top
```

Mechanically:
```
Original tensor → split into 16-element blocks → for each block:
  - Apply per-tensor FP32 scale
  - Apply per-block FP8 (E4M3) scale  
  - Round to FP4 (E2M1)
```

Why this wins:
- Smaller blocks (16 vs MXFP4's 32) → less averaging across diverse values
- FP8 scale type (vs MXFP4's E8M0 which is just a power-of-2 exponent) → finer granularity per block
- Two levels capture both global tensor stats and local block stats

The cost: more storage overhead (the scales). NVFP4 averages ~4.25 bits/weight including scales vs theoretical 4 bits.

## NVFP4 vs MXFP4 — the 2026 fault line

| | NVFP4 | MXFP4 (OCP) |
|---|---|---|
| Block size | 16 | 32 |
| Per-block scale type | FP8 E4M3 | E8M0 (power-of-2) |
| Global scale | FP32 per-tensor | None |
| Accuracy | Better at FP4 | Slightly worse, but improving with MR-GPTQ |
| Hardware | Blackwell only | Blackwell + AMD MI355 + multi-vendor future |
| When to use | NVIDIA-only deployment | Multi-vendor / portability |

Mental model: **NVFP4 is NVIDIA's "we'll squeeze every drop of accuracy" version. MXFP4 is the open standard for cross-vendor compatibility.** Both are valid for production in 2026; pick based on hardware lock-in tolerance.

## MXFP6, MXFP8

OCP defined a family of microscaling formats. In practice:

- **MXFP8** — block-scaled FP8. Used in some training setups; less common in inference because plain FP8 already works.
- **MXFP6** — exists, rarely used. Teams jump MXFP8 → NVFP4.
- **MXFP4** — covered above.

When you see MX-anything in a paper or release notes, default mental model: same idea as NVFP4 (block scaling), different parameters.

## The FP8 / FP4 hardware story

| Format | A100 | H100 / H200 | B100 / B200 | MI300X | MI355X |
|---|---|---|---|---|---|
| BF16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| FP8 (E4M3, E5M2) | ✗ | ✓ | ✓ | ✓ | ✓ |
| MXFP8 | ✗ | software | ✓ | software | ✓ |
| NVFP4 | ✗ | ✗ | ✓ | ✗ | ✗ |
| MXFP4 | ✗ | ✗ | ✓ | ✗ | ✓ |

**Critical rule**: a model quantized to NVFP4 *only runs fast on Blackwell*. Without native kernels you'll dequantize-on-the-fly to BF16 and lose all the speed.

Throughput math on B200:
```
BF16 dense:  ~2250 TFLOPS
FP8  dense:  ~4500 TFLOPS    (2× FP16)
FP4  dense:  ~9000 TFLOPS    (2× FP8)
```

## llm-compressor in 2026

`llm-compressor` v0.9 (Jan 2026) is the canonical recipe library for vLLM-aligned quantization. Big additions over 2024-2025:

- **Attention quantization** — not just linear layers. The KV cache projections get quantized too.
- **MXFP4 support** — added in v0.9.
- **Multi-compressor recipes** — non-uniform schemes in a single model. Example: NVFP4 attention + FP8 MLP.
- **Data-free PTQ for FP8 and NVFP4A16** — no calibration set needed for these recipes (newer; calibration still helps quality).
- **`static_minmax` is the default observer for NVFP4 activations** (was dynamic in earlier versions).

Typical recipe in 2026 (from llm-compressor docs):

```python
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",   # or "FP8_DYNAMIC", "MXFP4", "FP8_E4M3"
    ignore=["lm_head"],  # output projection often left at higher precision
)
```

Apply to model + push to HuggingFace. vLLM picks it up automatically.

## TorchAO

PyTorch's quantization library. In 2026:

- NVFP4 inference path matured (was experimental in 2025)
- Composes with `torch.compile` (had bugs through PyTorch 2.7; resolved in 2.8)
- Diffusers integration for image gen

For inference: `llm-compressor` is more mature for LLMs. TorchAO is better for non-LLM (diffusion, audio) and for research where you want to compose with torch.compile and FSDP.

## What you'll actually do

Three measurements continuing from Topic 01:

1. **FP8 W8A8** via llm-compressor or vLLM's built-in FP8 — measure throughput, memory, quality
2. **NVFP4** (if you have Blackwell) — same
3. **MXFP4** (Blackwell or MI355) — same

Add to your quality-vs-cost table. The numbers from this topic should *dominate* — FP8 is the 2026 inference default for a reason.

## Pitfalls

1. **Treating "FP8" as one thing.** E4M3 vs E5M2; per-tensor vs per-row scaling; W8A8 vs W8A16. State all three.
2. **Calibration set domain mismatch.** Calibrate on chat data, deploy on code → quality regression invisible until users complain. Use representative calibration data.
3. **Forgetting the activation regime.** Outliers in activations can blow per-tensor scales. SmoothQuant (Topic 03) preprocesses activations to make them more quantization-friendly.
4. **Comparing NVFP4 throughput on Hopper vs BF16 on Blackwell.** Always compare same-hardware-different-precision *or* same-precision-different-hardware. Don't mix.
5. **Skipping quality measurement.** The whole point of Topic 06.

## References

- NVIDIA: Introducing NVFP4 — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Pretraining LLMs with NVFP4 (arXiv) — https://arxiv.org/html/2509.25149
- MXFP4 on GPU Cloud (Spheron) — https://www.spheron.network/blog/mxfp4-microscaling-quantization-gpu-cloud/
- LLM Compressor v0.9 release notes — https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more
- LLM Compressor NVFP4 recipe — https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/
- NVIDIA Transformer Engine FP8 primer — https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- TorchAO + Diffusers Blackwell post — https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
