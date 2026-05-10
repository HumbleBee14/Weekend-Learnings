# 13 — ONNX Runtime and TensorRT (the runtime)

## Files

- `CONCEPTS.md` — what ORT and TRT are (vs TRT-LLM), the decision tree, the production-stack picture
- `bench_embedding.py` — same embedding model across PyTorch eager, `torch.compile`, ORT CPU/CUDA/TRT EPs

## Quickstart

```bash
pip install torch transformers "optimum[onnxruntime]"
pip install onnxruntime          # CPU
pip install onnxruntime-gpu      # CUDA + TRT EP

python bench_embedding.py
```

## Expected output (rough shape)

```
Benchmarking BGE-small across runtimes (embeddings/sec)
  seq_len=128  warmup=5  measure=50

runtime                       1          8         32        128
pytorch eager               980       6300      19500      52000
pytorch compile            1450       9100      27000      72000
ORT CPU                     820       4200      11000      18000
ORT CUDA                   1700      11000      35000      95000
ORT TensorRT               2100      14000      48000     140000
```

The TRT-EP advantage grows with batch size. At batch 1 on a small model, raw PyTorch eager is often surprisingly close to ORT — overhead per call dominates.

## Try

- **Repeat with a ViT** (`google/vit-base-patch16-224`). The TRT win should be larger because vision shapes are static.
- **CPU-only with `torch.compile`** — recent PyTorch CPU paths have improved; sometimes within 20% of ORT-CPU.
- **Quantize the ONNX model** (`optimum-cli onnxruntime quantize`). INT8 ORT on CPU is often the cheapest embedding-server config.
- **Wire ORT into a real serving stack** — Triton Inference Server (NVIDIA) accepts both ONNX and TRT plans; one Triton instance can serve a vLLM-side LLM and an ORT-side embedding model.

## Where this goes

- A real production stack is several runtimes — vLLM for the LLM, ORT or TRT for the small models around it. Knowing where each wins is part of platform design.
- Topic 14 — VLM serving has the same shape: ORT/TRT for the vision encoder, vLLM for the LLM decoder.
