# 06 — Apple Neural Engine and Core ML

## What the ANE actually is

A fixed-function neural inference accelerator on Apple Silicon, separate from the GPU. ~16 cores on M-series. Optimized for INT8 and FP16 dense matmul/conv with fixed shapes. Two key facts:

1. It is not addressable from MLX or Metal directly. The only legal path is Core ML.
2. It excels at fixed-shape models with low latency requirements. It is awkward for variable-length sequences, dynamic KV caches, attention masks — i.e. the modern LLM decode loop.

```
  +--------------------+      +-------------------+      +----------------+
  |  CPU (P + E cores) |  ==> |  Unified DRAM     | <==  |  GPU (Metal)   |
  +--------------------+      |  (zero-copy)      |      +----------------+
                              |                   |
                              |                   |  <==  |  ANE (Core ML) |
                              +-------------------+      +----------------+
```

The ANE shares the same DRAM as everything else (UMA, Topic 01), so transfer between CPU/GPU/ANE-resident tensors is free. The cost lives in the model graph rewrite Core ML does to land ops on ANE.

## Core ML in 2026

Core ML is Apple's deployment runtime. It accepts a model graph (`.mlpackage`), partitions it across CPU+GPU+ANE, and runs it. You author models in PyTorch / TensorFlow / JAX, then convert with `coremltools`.

Key 2026 features:

- `MLComputeUnits.cpuAndNeuralEngine`, `.cpuAndGPU`, `.all` — explicit dispatch hint.
- Quantization: weight palettization (1–8 bit), activation quantization, post-training and QAT.
- KV cache support via stateful models (`MLState`) — added iOS 18 / macOS 15. KV is now first-class, not a 2D-array hack.
- Mixed-bit palettization (2/4/6) — `coremltools.optimize.coreml`.
- Prediction options: `MLPredictionOptions.usesCPUOnly` is deprecated; use `MLModelConfiguration.computeUnits`.

## Where Core ML wins in 2026

- **iPhone deployment.** ANE runs at much lower power than GPU. For an app that runs a small model many times per second (Vision tasks, Whisper for dictation, on-device embedding), ANE is the right substrate.
- **Stable Diffusion / SDXL / FLUX on iPhone.** Apple's `ml-stable-diffusion` repo packages these for ANE+GPU. Practical 1–3s/image on A17/A18.
- **System integrations.** Vision framework, Speech framework, CreateML, NaturalLanguage — all driven by Core ML under the hood. If you want your model to integrate with these, Core ML is the only option.
- **Background execution.** ANE inference does not preempt GPU rendering — useful when running ML alongside graphics.

## Where MLX wins

- LLM serving on Mac. Variable-length sequences and a growing KV cache are exactly the workload Core ML's static-shape origins handle worst.
- Iteration speed. Pythonic, no compile-and-deploy cycle.
- Anything that touches research code unchanged.

## The LLM-on-ANE story is awkward but real

Core ML did add `MLState` for KV cache, and Apple ships `swift-transformers` with examples. Llama-2-7B 4-bit on ANE-aware Core ML hits roughly 8–15 tok/s on M3. That is much slower than MLX (200+ tok/s) on the same machine. The reasons:

- ANE wants fixed shapes. KV cache that grows by one row per token forces re-compilation or padding to a fixed maximum context, which wastes compute.
- ANE INT8 paths are well-tuned; INT4 is via palettization, which decompresses on the fly and loses some of the bandwidth win.
- The decode loop is bandwidth-bound, not compute-bound. ANE's compute advantage matters less here than the GPU's higher memory bandwidth.

Translation: do not pick ANE to serve a 7B chat model on a Mac. Pick MLX. Pick ANE on iPhone where GPU power budget is tight.

## Conversion flow

```
  PyTorch model (eager)
       |
       v
  torch.jit.trace OR torch.export
       |
       v
  coremltools.convert(...)
       |
       v
  .mlpackage  (graph + weights, signed)
       |
       v
  Xcode / CoreML.framework
```

```python
import torch, coremltools as ct

model = MyModel().eval()
example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=example.shape)],
    compute_units=ct.ComputeUnit.CPU_AND_NE,   # ANE-eligible
    minimum_deployment_target=ct.target.iOS18,
)
mlmodel.save("MyModel.mlpackage")
```

After converting, profile with **Xcode > Instruments > Core ML** — this shows per-op dispatch (CPU vs GPU vs ANE). The number you care about: what fraction of compute landed on ANE. If it is < 80%, your model has ops the ANE compiler refused, and you are paying transfer cost between units.

## Common ANE-unfriendly ops in 2026

- Dynamic shapes (variable batch, variable seq) — use `RangeDim` carefully or build static-shape variants.
- Custom ops, anything outside the ML Program op set.
- Some attention variants. Standard scaled-dot-product is fine; novel masking patterns may fall back.
- 64-bit integer indexing.

`coremltools.utils.MultiFunctionDescriptor` (2025+) lets you pack multiple input shapes into one `.mlpackage` — the runtime picks the closest match. This is the modern answer to dynamic-shape ANE.

## Common pitfalls

1. **Assuming `compute_units=.all` lands on ANE.** It only enables ANE eligibility. The compiler still decides per-op. Always profile.
2. **Forgetting `minimum_deployment_target`.** Older targets miss `MLState` and modern palettization. Set iOS 18 / macOS 15 minimum unless you must support older devices.
3. **Comparing Core ML LLMs to MLX on Mac.** Wrong axis — ANE is for iPhone, MLX is for Mac dev. Pick by device, not by benchmark.
4. **Quantizing post-trace with PyTorch tooling.** Use `coremltools.optimize.coreml` — it produces ANE-compatible palettized weights. PyTorch INT8 quantization graphs frequently do not lower cleanly.

## References

- Core ML docs: https://developer.apple.com/documentation/coreml
- coremltools: https://github.com/apple/coremltools
- ml-stable-diffusion: https://github.com/apple/ml-stable-diffusion
- swift-transformers: https://github.com/huggingface/swift-transformers
- ANE deployment guide (Apple): https://machinelearning.apple.com/research/neural-engine-transformers
- Core ML stateful models: https://developer.apple.com/documentation/coreml/mlstate
- Mixed-bit palettization: https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html
