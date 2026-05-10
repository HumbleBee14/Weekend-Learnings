# 05 — llama.cpp Deep Dive

## Files

- `CONCEPTS.md` — GGML / GGUF, the quant family in 2026, where it wins / loses, when CPU beats GPU
- `bench_local.py` — OpenAI-client benchmark pointed at `llama-server`

## Quickstart

```bash
brew install llama.cpp     # Mac; for CUDA/Vulkan, build from source

huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ./models

llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8003 --ctx-size 8192 -ngl 999 --parallel 8

# in another shell
python bench_local.py --concurrency 1
python bench_local.py --concurrency 8
```

## Expected output (rough shape)

On an M3 Max at batch=1, 7B Q4_K_M:

```
llama.cpp via http://localhost:8003/v1
  agg throughput    65 tok/s
  TTFT p50/p95/p99  140 / 260 / 320  ms
```

On an L4 at batch=8, 7B Q4_K_M:

```
agg throughput    420 tok/s
TTFT p50/p95/p99  180 / 380 / 500  ms
```

vLLM on the same L4 with FP8 will do 1500+ tok/s at batch=8 — that's the trade.

## Try

- **Same workload at concurrency 1, then 16.** Watch llama.cpp degrade more steeply than vLLM did in Topic 01 — continuous batching is less mature.
- **CPU-only mode (`-ngl 0`).** Compare $/Mtok against a small GPU. This is the cross-substrate cost run from Project 2's break-it list.
- **Try i-quants:** download `IQ4_XS` instead of `Q4_K_M`. Smaller file, similar quality. Compare throughput.
- **Try Unsloth Dynamic v2.0** — best-quality-per-bit at small sizes.
- **Use `llama-bench`** (the official microbench) for kernel-level numbers — different from end-to-end serving numbers.

## Where this goes

- Topic 07 — llama.cpp is one of the bake-off engines (and the only one that competes on CPU)
- Level 8 — Apple Silicon agents lean on llama.cpp's Metal backend
- Project 2 G9 (cost per million tokens) explicitly includes a CPU-only llama.cpp row
