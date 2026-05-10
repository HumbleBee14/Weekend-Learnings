# 06 — ANE and Core ML

## Files

- `CONCEPTS.md` — what the ANE is, where Core ML wins vs MLX, conversion flow, the LLM-on-ANE awkwardness, common pitfalls.
- `convert_to_coreml.py` — converts a small PyTorch vision model to a Core ML `.mlpackage` with ANE eligibility, then runs it through Core ML on macOS.

## Quickstart

```bash
pip install torch torchvision coremltools numpy
python convert_to_coreml.py
```

This produces `ResNet18.mlpackage` and runs one inference on it via Core ML, printing the top-5 ImageNet indices.

## Expected output

```
Tracing model...
Converting to Core ML (compute_units = CPU_AND_NE)...
Saved ResNet18.mlpackage
Core ML output top-5 indices: [281, 285, 282, 287, 283]
```

The actual indices depend on the random input — the point is the conversion + on-device inference path runs end-to-end.

## Try

- Open `ResNet18.mlpackage` in Xcode and inspect the *Performance* tab. It will show predicted dispatch across CPU / GPU / ANE per op. Aim for >80% ANE.
- Re-convert with `compute_units=ct.ComputeUnit.CPU_AND_GPU` and compare predicted latency in Xcode's perf report.
- Apply palettization: `from coremltools.optimize.coreml import palettize_weights, OpPalettizerConfig` — drop weights to 4-bit and re-measure size and dispatch.
- Swap in a tiny transformer block (e.g. a 2-layer GPT-2-shape model). Watch how many ops fall back to CPU/GPU because of attention masking. This is exactly why LLMs on ANE is awkward.

## Where this goes

Topic 07 is the higher-level Apple-native path — Foundation Models — which uses Core ML and the ANE under the hood but hides the conversion entirely. Topic 11 builds the agentic loop on MLX, where ANE is intentionally not in the picture for LLM decode.
