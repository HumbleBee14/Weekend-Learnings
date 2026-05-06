# 06 — Local-First Touch

## Files

- `CONCEPTS.md` — what llama.cpp and Ollama are, GGUF, quantization, why they're built differently
- `compare.py` — sends the same prompt to your FastAPI server and to Ollama, prints latency + tok/s side by side

## Quickstart

```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.com/download

# Start the daemon (runs in background on port 11434)
ollama serve &

# Pull the same model family you've been using
ollama pull qwen2.5:0.5b

# Run interactively
ollama run qwen2.5:0.5b "Explain merge sort"

# Or hit the API directly
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen2.5:0.5b", "prompt": "Hi", "stream": false}'

# Side-by-side comparison (start your topic-03 server first)
python compare.py
```

## Reading the comparison

You'll see something like:

```
FastAPI + Qwen 0.5B FP16:    1820ms, 80 tok, 43.9 tok/s
Ollama + qwen2.5:0.5b Q4_K_M: 750ms, 80 tok, 106.6 tok/s
```

Ollama wins on speed because:
- 4-bit quantization → 4× less memory bandwidth needed
- Hand-tuned C++ + SIMD kernels
- mmap'd weights, no Python overhead
- Cold start in milliseconds vs your server's seconds

Your server can win at:
- Higher concurrency (Ollama serializes requests heavily)
- Custom logic (your code, not theirs)
- Models without good GGUF conversions yet
- Datacenter GPUs where the FP16 → quant gap is smaller

## What to take away

There's no universal best inference stack. Datacenter GPUs + FP16 + vLLM is one regime. Laptops + 4-bit GGUF + llama.cpp is another. Both are real production paths in 2026.

You'll meet llama.cpp again in Level 5 (engine bake-off — it's one of the four engines compared) and Level 8 (full local-first week with MLX + llama.cpp on Apple Silicon).
