# 06 — Local-First Touch (Ollama + llama.cpp)

## Why a detour into local serving now

The whole curriculum is about "datacenter" inference — Python + PyTorch + GPUs + cloud. Half the field runs differently: small models, on a laptop, no GPU, often quantized to 4 bits. Tools: **Ollama** and **llama.cpp**.

Two hours of contact with these now. They come back hard in Level 5 (engine bake-off) and Level 8 (full local-first week). Get a feel now while expectations are low.

## What llama.cpp is

C++ implementation of LLaMA-style transformer inference, optimized for CPU and Apple Silicon. Originally Georgi Gerganov's hobby project; now the de facto standard for "run an LLM on consumer hardware."

Key architectural facts:
- **GGUF format** — a single-file format combining weights, tokenizer, metadata. Quantized (Q4_K_M, Q5_K_M, Q8_0, etc.) — most people use Q4_K_M as the practical default.
- **Memory-mapped weights** — `mmap()` the GGUF file. The OS pages it in lazily; never loads the full thing into RAM unless you actually use it.
- **No Python in the hot path** — everything is C++ + SIMD + Metal/Vulkan/CUDA.
- **Metal backend on Apple Silicon** — uses Apple's GPU compute API. Decent throughput on M-series chips.
- **HTTP server included** — `llama.cpp` ships an OpenAI-compatible server. You can use it like vLLM.

Why it's fast on CPU: hand-tuned SIMD kernels (AVX2, AVX-512, NEON, AMX). For small quantized models, CPU can be competitive with low-end GPUs.

## What Ollama is

A wrapper *around* llama.cpp. Adds:
- Easy model pulling (`ollama pull qwen2.5:0.5b`)
- Modelfile (Dockerfile-like config for prompt templates, system messages, etc.)
- Background daemon (`ollama serve`) with an HTTP API
- Automatic GPU offload detection
- A growing model library

In 2026 Ollama also has an MLX backend on Apple Silicon — for some workloads MLX is 2× faster than llama.cpp's Metal backend.

You'd use Ollama for: easy local development, prototyping, agentic IDE backends. You'd use raw llama.cpp for: maximum control, custom builds, production embedding in another C++ app.

## How they compare to your FastAPI server

Same job (HTTP → completions). Different tradeoffs:

| | Your FastAPI server | Ollama / llama.cpp |
|---|---|---|
| Python runtime | yes | no |
| Quantization | none yet | aggressive (4-bit default) |
| Cold start | seconds | milliseconds (mmap) |
| Memory footprint | full FP16 weights | quantized GGUF |
| Best at | datacenter GPUs, big models | laptops, small models |
| Code to write to add a new model | tokenizer + chat template + maybe more | `ollama pull <name>` |

The lesson: there's no single right way to serve an LLM. The hardware and the use case decide.

## What to actually do

1. Install Ollama: `brew install ollama` on Mac, or [download](https://ollama.com/download) for your OS.
2. Pull a small model: `ollama pull qwen2.5:0.5b`
3. Run it: `ollama run qwen2.5:0.5b "Explain merge sort"`
4. Check the API: `curl http://localhost:11434/api/generate -d '{"model": "qwen2.5:0.5b", "prompt": "Hi"}'`
5. Compare: same prompt to your FastAPI server, same prompt to Ollama. Note TTFT and tokens/sec for both.

Don't over-analyze. The goal is a feel, not a benchmark.

## Pitfalls

1. **Comparing FP16 (your server) to Q4_K_M (Ollama) and calling them equal.** They're different precisions. The quality is similar but not identical; the speed is dramatically different.
2. **Treating Ollama like vLLM.** It's not built for high concurrency. It serves one request at a time well; under heavy load it falls apart. That's not its job.
3. **Forgetting that Ollama's API is *not* OpenAI-compatible by default.** It uses its own schema. Use the `/v1/chat/completions` endpoint if you need OpenAI compatibility.
