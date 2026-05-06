"""
Side-by-side: same prompt against your FastAPI server (topic-03) and Ollama.

Setup:
  - Topic-03 batched server running on http://localhost:8000
  - Ollama running on http://localhost:11434 with qwen2.5:0.5b pulled

Run:
    ollama serve &  # if not running
    ollama pull qwen2.5:0.5b
    python compare.py
"""

import asyncio
from time import perf_counter

import httpx

YOUR_URL = "http://localhost:8000/generate"
OLLAMA_URL = "http://localhost:11434/api/generate"
PROMPT = "Explain how a hash table works in three sentences."


async def hit_your_server(client):
    t0 = perf_counter()
    resp = await client.post(YOUR_URL, json={"prompt": PROMPT, "max_tokens": 80}, timeout=120)
    elapsed = (perf_counter() - t0) * 1000
    body = resp.json()
    return {
        "stack": "FastAPI + Qwen 0.5B FP16",
        "latency_ms": elapsed,
        "tokens": body["tokens_generated"],
        "tokens_per_sec": body["tokens_generated"] / (elapsed / 1000),
        "completion": body["completion"][:80],
    }


async def hit_ollama(client):
    t0 = perf_counter()
    # Ollama's /api/generate streams by default; we set stream=False for a one-shot response
    resp = await client.post(
        OLLAMA_URL,
        json={"model": "qwen2.5:0.5b", "prompt": PROMPT, "stream": False, "options": {"num_predict": 80}},
        timeout=120,
    )
    elapsed = (perf_counter() - t0) * 1000
    body = resp.json()
    # Ollama returns eval_count = generated tokens, eval_duration = ns
    tokens = body.get("eval_count", 0)
    return {
        "stack": "Ollama + qwen2.5:0.5b Q4_K_M",
        "latency_ms": elapsed,
        "tokens": tokens,
        "tokens_per_sec": tokens / (elapsed / 1000) if tokens else 0,
        "completion": body.get("response", "")[:80],
    }


async def main():
    async with httpx.AsyncClient() as client:
        # Warmup
        try:
            await hit_your_server(client)
        except Exception as e:
            print(f"Your server unreachable: {e}")
        try:
            await hit_ollama(client)
        except Exception as e:
            print(f"Ollama unreachable: {e}")

        print(f"Prompt: {PROMPT}\n")

        for run in [1, 2, 3]:
            print(f"--- run {run} ---")
            try:
                a = await hit_your_server(client)
                print(f"  {a['stack']}: {a['latency_ms']:.0f}ms, {a['tokens']} tok, {a['tokens_per_sec']:.1f} tok/s")
                print(f"    {a['completion']!r}")
            except Exception as e:
                print(f"  Your server: error: {e}")
            try:
                b = await hit_ollama(client)
                print(f"  {b['stack']}: {b['latency_ms']:.0f}ms, {b['tokens']} tok, {b['tokens_per_sec']:.1f} tok/s")
                print(f"    {b['completion']!r}")
            except Exception as e:
                print(f"  Ollama: error: {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
