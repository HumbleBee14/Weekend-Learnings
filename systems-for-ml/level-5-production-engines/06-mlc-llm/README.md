# 06 — MLC-LLM

## Files

- `CONCEPTS.md` — what TVM Unity does, when you reach for MLC, the WebGPU differentiator
- `compile_and_run.py` — minimal MLCEngine chat loop using a pre-compiled artifact

## Quickstart

The single most valuable thing here is opening **https://chat.webllm.ai/** in Chrome and watching a 7B run in your browser via WebGPU. Send a few prompts, inspect dev tools, confirm zero network calls after the model download.

For a local CLI run:

```bash
pip install --pre mlc-llm-nightly mlc-ai-nightly
python compile_and_run.py
```

## Expected output

```
Loading HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC ...
  ready in 24.3s

>>> Why does MLC-LLM exist when vLLM and llama.cpp already do?
MLC compiles the model graph itself, so one source targets CUDA, Metal,
Vulkan, ROCm, and WebGPU without per-backend kernel code...

Total: ~380 tokens in 6.4s = 59 tok/s
```

Numbers vary widely by target. The point isn't peak throughput — it's that the same `compile_and_run.py` can target a phone, a browser, an AMD APU, and an H100 with a recompile.

## Try

- **Run the WebLLM demo on a friend's machine** with no GPU. Confirm it falls back gracefully (or doesn't load — also informative).
- **Compile a custom model** for your target via `mlc_llm convert_weight` and `mlc_llm compile`. Note the time; that's the operational cost.
- **Compare to llama.cpp on the same hardware.** llama.cpp is usually faster on CPU/Metal; MLC's value isn't peak speed.

## Where this goes

- Topic 07 — MLC isn't a bake-off entry by default; mention it as the "cross-platform" option in your engine recommendation
- Level 8 — for Apple Silicon agents, MLX or llama.cpp will usually win over MLC; MLC's lane is heterogeneous fleets
