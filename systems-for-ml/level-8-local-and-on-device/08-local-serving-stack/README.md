# 08 — Local Serving Stack

## Files

- `CONCEPTS.md` — vLLM-MLX vs Ollama vs LM Studio vs llama-server, the OpenAI-compatibility contract, when to pick which, multi-model patterns.
- `bench_serving.py` — hits any OpenAI-compatible endpoint with N concurrent requests; reports TTFT distribution, tokens/sec, completed/sec.
- `start_servers.sh` — convenience script to start Ollama (MLX backend) and `mlx_lm.server` side-by-side on different ports.

## Quickstart

```bash
# Pick one engine to start. Ollama with MLX backend:
brew install ollama
OLLAMA_BACKEND=mlx ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M

# Or mlx_lm directly:
pip install mlx-lm
python -m mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8000 &

# Then bench:
pip install openai httpx
python bench_serving.py \
    --base-url http://localhost:11434/v1 \
    --model qwen2.5:7b-instruct-q4_K_M \
    --concurrency 8 \
    --requests 32
```

## Expected output

```
Engine: http://localhost:11434/v1  model=qwen2.5:7b-instruct-q4_K_M
Concurrency=8  Requests=32
TTFT  p50=180ms  p90=320ms  p99=420ms
Output tok/s aggregate: ~395
Wall time: 21.4s    Completed: 32/32
```

Numbers depend heavily on the engine and hardware. Ollama (no continuous batching) will have aggregate tok/s only modestly above the single-request number. vLLM-MLX with the same concurrency should comfortably hit 400+ tok/s aggregate.

## Try

- Start the same model in two engines on two ports and run `bench_serving.py` against each. Capture the gap. This is part of **G18 of Project 4**.
- Increase `--concurrency` to 16, 32. Watch where each engine's TTFT distribution explodes.
- Swap `--model` to a 4-bit MoE (Topic 09) and see how aggregate throughput changes — MoE active-bandwidth math kicks in.
- Add `--prompt-tokens 8000` to stress prefill. TTFT becomes the dominant cost; memory pressure is more interesting (G19).

## Where this goes

Topic 09 is the MoE story for these same engines — Llama 4 Scout and Qwen3-Next change the math. Topic 11 builds the agentic loop that will hit one of these endpoints under load.
