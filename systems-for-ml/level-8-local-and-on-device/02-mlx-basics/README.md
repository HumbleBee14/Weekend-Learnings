# 02 — MLX Basics

## Files

- `CONCEPTS.md` — MLX's lazy graph, unified-memory-native arrays, the mlx-lm / mlx-vlm / mlx-whisper / mlx-embeddings ecosystem, dtypes, autograd, common pitfalls.
- `hello_mlx.py` — lazy/eval demo, autograd via `mx.grad`, matmul throughput probe, tiny MLP forward.

## Quickstart

```bash
pip install mlx mlx-lm
python hello_mlx.py
```

Run a real model end-to-end:

```bash
python -m mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Explain unified memory in one paragraph." \
  --max-tokens 200
```

## Expected output

`hello_mlx.py` prints the lazy/eager comparison, gradients of `x^3 + 2x` at `[1,2,3]` = `[5, 14, 29]`, and a TFLOPS number for fp16 matmul. On M3 Max expect ~14–18 TFLOPS effective on the 4096^2 fp16 matmul. M5 Max with Neural Accelerators is ~3–4x higher on the same call (Topic 04).

## Try

- Quantize a model yourself: `python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./q4 -q --q-bits 4`. Watch the disk size drop from ~14GB to ~4GB.
- Replace the matmul with `mx.fast.matmul` and re-time on M5 hardware (Topic 04 covers what changes).
- Add `mx.eval(c)` *inside* the timing loop instead of after — observe the per-op overhead and why lazy fusion matters.
- Swap `mx.float16` for `mx.bfloat16`. Same TFLOPS on most hardware; slightly different numerics.

## Where this goes

Topic 03 lines MLX up next to `llama.cpp` Metal and PyTorch MPS on the same model and measures the gap. Topic 04 is the M5 Neural Accelerator path. Topic 12 fine-tunes via `mlx_lm.lora`, which is built on the same primitives shown here.
