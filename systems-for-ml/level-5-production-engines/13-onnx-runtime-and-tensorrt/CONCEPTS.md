# 13 — ONNX Runtime and TensorRT (the runtime, not TRT-LLM)

This is the non-LLM-specific runtime path. Two production runtimes that exist *outside* the vLLM/SGLang/TRT-LLM stack and that you'll meet in real codebases. They are not optimized for autoregressive LLM serving — but they're often the right choice for embeddings, reranking, vision, audio, classification, and the small-transformer surface area around the LLM.

## ONNX Runtime (ORT)

Microsoft's cross-framework inference runtime. Takes an ONNX-format model (an open exchange format) and runs it via a graph executor.

```
PyTorch / TF / JAX model
        │
        │  torch.onnx.export, tf2onnx, jax2tf, optimum.exporters
        ▼
.onnx file (graph + weights, framework-agnostic)
        │
        │  ORT loads, applies graph optimizations (constant folding,
        │  layer fusion, dead-code elim, layout transforms)
        ▼
Execution Provider dispatch:
   ┌──────────────────────────────────────────────────────────────┐
   │  CPU (default — uses oneDNN / MLAS / AVX-512 / AMX)          │
   │  CUDA EP                                                      │
   │  ROCm EP                                                      │
   │  TensorRT EP   ← ORT calls TRT for the layers TRT can do      │
   │  CoreML EP     (Apple)                                        │
   │  DirectML EP   (Windows)                                      │
   │  WebGPU EP     (browsers)                                     │
   │  OpenVINO EP   (Intel)                                        │
   │  QNN EP        (Qualcomm NPU)                                 │
   └──────────────────────────────────────────────────────────────┘
```

The thesis: one model file, one runtime API, many backends. Cross-platform portability that PyTorch eager can't match.

### Where ORT wins

- **Embedding / reranking models at scale.** BGE, jina-embeddings, ColBERT, Cohere reranker, BERT-family. ORT's graph optimizer beats raw PyTorch on these by 2-3× and ships as a static binary.
- **CPU inference.** Small classification / NER models on CPU clusters (still common in production for cost reasons).
- **Edge deployment.** Windows, Android, browsers (via WebGPU). PyTorch can't go there cleanly.
- **Cross-framework portability.** Train in PyTorch or TF, export once to ONNX, run anywhere.
- **Older transformer surfaces.** Models that don't have a vLLM kernel path — small encoders, cross-encoders, dense retrievers.

### Where ORT loses

- **Autoregressive LLM serving.** No paged KV, no continuous batching. You'd write that on top, badly. Use vLLM/SGLang/TRT-LLM.
- **Hopper / Blackwell-specific kernels.** TRT-LLM and vLLM's tensor-core paths are usually faster for big LLMs.
- **Bleeding-edge model architectures.** Anything that needs new attention variants (MLA, sliding window with sinks, native multimodal) lags ORT support.

## TensorRT (the runtime)

NVIDIA's inference compiler+runtime, NVIDIA-only. Distinct from TRT-LLM. Takes a model (ONNX, or via the TRT API) and produces an **engine plan** — a hardware-and-version-specific binary — that runs through the TRT runtime.

```
ONNX or PyTorch model
       │
       │  trtexec / Python TRT API
       ▼
Engine plan (.plan / .engine file)
       │
       │  pinned to: GPU SM version, CUDA version, TRT version
       ▼
TRT runtime loads .plan, executes
       │
       │  features: kernel auto-tuning per shape, layer fusion,
       │            INT8/FP8 calibration, layout selection,
       │            cuDNN/cuBLAS/myelin op fallbacks
       ▼
output
```

### Where TRT (not TRT-LLM) wins

- **Vision models.** ResNet, ViT, DETR, YOLO, SAM. TRT plans are 2-5× faster than PyTorch eager and slightly faster than ORT.
- **Audio / ASR.** Whisper, Conformer. Static-shape encoder-decoder fits TRT well.
- **Sub-billion-parameter transformers** with fixed shapes — fits TRT's compiled-plan model well.
- **Embedded NVIDIA** (Jetson, Orin). TRT is the standard runtime there.

### Where TRT loses

- **Dynamic-shape autoregressive workloads.** TRT prefers static shapes; LLM decode has dynamic sequence length. This is *exactly* why TRT-LLM exists — to handle the LLM-specific dynamics on top of TRT.
- **Fast iteration.** Re-building plans takes minutes; bad for development loops.
- **Cross-platform.** NVIDIA-only.

## The decision tree

```
Is your workload autoregressive LLM serving?
  ├── Yes → vLLM / SGLang / TRT-LLM (Topics 01-04)
  └── No
       │
       └── Is it transformer-shaped (encoder, classifier, ViT, ASR)?
            ├── Yes
            │    └── On NVIDIA + need max throughput?
            │         ├── Yes → TensorRT (or ORT-with-TRT-EP)
            │         └── No  → ONNX Runtime
            └── No (general DL: MLP, CNN, mixed)
                 └── ONNX Runtime (CPU EP or CUDA EP)
```

## Why this is in the LLM-engines week

A real production stack is *several* runtimes:

```
                         ┌─────────────────────────┐
                         │ Big LLM serving          │
                         │  vLLM / SGLang           │
                         └─────────────────────────┘
                                    │
                                    │ retrieves context from
                                    ▼
                         ┌─────────────────────────┐
                         │ Embedding model          │
                         │  ORT on CPU or CUDA EP   │
                         └─────────────────────────┘
                                    │
                                    │ ranked by
                                    ▼
                         ┌─────────────────────────┐
                         │ Cross-encoder reranker   │
                         │  ORT or TensorRT         │
                         └─────────────────────────┘
                                    ▲
                                    │ filtered by
                                    │
                         ┌─────────────────────────┐
                         │ Safety classifier        │
                         │  ORT on CPU              │
                         └─────────────────────────┘
                                    ▲
                                    │ for VLM input
                                    │
                         ┌─────────────────────────┐
                         │ Vision encoder (ViT/SigLIP)│
                         │  TensorRT for max thru.  │
                         └─────────────────────────┘
```

Each component picks its runtime. If you only know vLLM, you have a hammer for every nail and you'll over-pay on the small-model tier.

## Pitfalls

1. **Trying to serve a 70B LLM on ORT.** It works (slowly). Don't.
2. **TRT plans not portable.** Plan built on H100 won't run on L4 (different SM). Build per target.
3. **Forgetting `torch.compile` exists.** For PyTorch-native deployments, `torch.compile` + Inductor often gets close to ORT on smaller models without leaving the Python ecosystem.
4. **Not using the TRT EP under ORT.** ORT-with-TRT-EP gives you ORT's portability with TRT's speed for the layers TRT supports. Best of both for vision pipelines.
5. **Static-shape assumptions that don't hold.** A "fixed-shape" embedding model with variable-length input still has dynamic shape. Read the export carefully.

## What to do this topic

1. Take a small embedding model (`BAAI/bge-small-en-v1.5`). Export to ONNX via `optimum`.
2. Run it via:
   - PyTorch eager
   - PyTorch + `torch.compile`
   - ORT CPU EP
   - ORT CUDA EP
   - ORT TensorRT EP (if you have GPU + TRT installed)
3. Measure embeddings/sec at batch 1, 8, 32, 128. The crossover where each wins is informative.
4. Repeat with a small ViT (`google/vit-base-patch16-224`). The TRT win should be larger.

## References

- ONNX Runtime — https://onnxruntime.ai/
- ORT execution providers — https://onnxruntime.ai/docs/execution-providers/
- TensorRT — https://docs.nvidia.com/deeplearning/tensorrt/
- Hugging Face Optimum (PyTorch → ONNX) — https://huggingface.co/docs/optimum/
- ORT for transformers — https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models
- TRT-EP under ORT — https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- ONNX spec — https://onnx.ai/
