# 08 — Local Serving Stack

## The 2026 ranked picture

```
  + ----------------------------------------------------- +
  |                  client (OpenAI SDK)                  |
  + ----------------------------------------------------- +
                       |  HTTP /v1/chat/completions
                       v
  + -------------- + + ------------- + + ---------------- +
  |  vLLM-MLX      | |  Ollama (MLX) | |  LM Studio       |
  |  :8000         | |  :11434       | |  :1234           |
  |  cont. batch   | |  CLI daemon   | |  GUI-first       |
  |  paged KV      | |  multi-model  | |  multi-model     |
  |  400+ tok/s    | |  LRU eviction | |  best UX         |
  + -------------- + + ------------- + + ---------------- +
                       |
  + ---------------- + |
  |  llama-server    | <-- universal fallback (cross-platform)
  |  :8080  GGUF     |
  + ---------------- +
                       |
                       v
              + ----------------- +
              |  MLX / llama.cpp  |
              |  Metal kernels    |
              +-------------------+
                       |
                       v
              +-------------------+
              |  M-series GPU     |
              +-------------------+
```

All four expose the OpenAI Chat Completions schema. The differences are concurrency, model management, and UX.

## vLLM-MLX

A port of vLLM's V1 scheduler to the MLX backend. Same continuous batching, same paged KV cache (Levels 4 / 5), now on Apple Silicon.

**Why it matters.** Ollama and LM Studio do not have continuous batching. They serialize requests. For a single developer at a keyboard that is fine — for serving real concurrent traffic from a Mac to a small team, this is the only option that scales.

**Numbers (M3 Max 64 GB, Qwen2.5-7B 4-bit).**

| Setting | Throughput | TTFT |
|---|---|---|
| Single request | ~230 tok/s | 80 ms |
| 8 concurrent requests | ~400 tok/s aggregate | 250 ms |
| 16 concurrent requests | ~480 tok/s aggregate | 450 ms |

Continuous batching wins as soon as you have more than two simultaneous users.

```bash
pip install vllm-mlx
python -m vllm.entrypoints.openai.api_server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --port 8000
```

## Ollama (MLX backend, March 2026 preview)

The CLI-first daemon a lot of people already have installed. Historical default backend was llama.cpp (GGUF). The MLX backend was added as a preview in March 2026 and is rolling toward default.

**Why it matters.** Same Ollama UX (`ollama pull`, `ollama run`, `ollama serve`) but ~50% faster on Apple Silicon by switching engines. OpenAI-compatible HTTP on `:11434`. Multi-model concurrent serving with LRU eviction — hold two models in memory until one ages out.

```bash
brew install ollama
OLLAMA_BACKEND=mlx ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b-instruct-q4_K_M",
       "messages":[{"role":"user","content":"hello"}]}'
```

Limitation: no continuous batching. Two simultaneous requests run sequentially.

## LM Studio

GUI-first. Closed-source app, free for personal use. Best onboarding of any local LLM tool — drag a model in, click serve, point an OpenAI client at `:1234`. MLX backend toggle in settings.

Use when: handing a non-engineer a working local stack, or you're exploring many models without committing to a CLI workflow.

## llama-server

The HTTP server bundled with `llama.cpp`. GGUF only. OpenAI-compatible. Cross-platform — runs on Linux/Windows/Mac/RPi/Android.

```bash
./llama-server -m qwen2.5-7b-q4.gguf --port 8080 -ngl 99
```

Use when you need to ship a single static binary, or you're on non-Apple hardware. The fallback that always works.

## OpenAI compatibility — the unifying contract

All four implement (most of) `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`. Practical implication: write your client against `openai-python` once, swap `base_url` to switch engines.

```python
from openai import OpenAI

vllm   = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="-")
lmstud = OpenAI(base_url="http://localhost:1234/v1", api_key="-")
```

Differences live in: tool-call schema strictness, structured-output guarantees, streaming chunk shape edge cases. Test each against your actual client before relying on parity.

## Choosing an engine — practical rubric

- Single dev at a keyboard, want the fastest decode → **MLX directly via `mlx_lm.server`** or vLLM-MLX.
- Multi-model exploration / shareable to a teammate without engineering setup → **LM Studio**.
- A CI script or background daemon that pulls models on demand → **Ollama**.
- Serving 5+ concurrent users from a Mac → **vLLM-MLX**.
- Cross-platform single binary → **llama-server**.

## Multi-model patterns

Local serving rarely runs one model. A typical agentic stack (Topic 11) wants:

- A small fast coder for autocomplete (Qwen2.5-Coder-1.5B).
- A medium chat model for the agent loop (Qwen2.5-7B).
- An embedding model for retrieval.

Ollama and LM Studio handle this with LRU eviction; vLLM-MLX runs one model per process so you launch separate servers and route at the client.

```
  +---------+    autocomplete     +-------------------+
  | client  | ------------------> | :11435  Coder-1.5 |
  |         |     chat            +-------------------+
  |         | --------------+
  |         |     embed     |     +-------------------+
  +---------+ -----------+  +---> | :11434  Qwen-7B   |
                         |        +-------------------+
                         |
                         |        +-------------------+
                         +------> | :11436  bge-m3    |
                                  +-------------------+
```

## Common pitfalls

1. **Pretending Ollama scales like vLLM.** It does not. No continuous batching. Use vLLM-MLX when concurrency > 2.
2. **Loading two big models in memory by accident.** Ollama's LRU keeps both around until pressure forces eviction. On a 32 GB Mac, two 7Bs at 4-bit will swap. Keep the keep-alive low.
3. **Skipping `--ngl 99` on llama-server.** Default is `0` (CPU). Free GPU left on the table.
4. **Mixing OpenAI tool-call dialects.** vLLM follows OpenAI strictly; Ollama is looser. Test before relying on schema.
5. **Forgetting the OS file-descriptor cap.** Concurrent streaming clients eat fds. `ulimit -n 4096`.

## References

- vLLM-MLX: https://github.com/waybarrios/vllm-mlx
- vLLM upstream: https://github.com/vllm-project/vllm
- Ollama: https://ollama.com / https://github.com/ollama/ollama
- Ollama MLX preview blog: https://ollama.com/blog/mlx
- LM Studio: https://lmstudio.ai
- llama.cpp server: https://github.com/ggerganov/llama.cpp/tree/master/examples/server
- mlx-lm server: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
