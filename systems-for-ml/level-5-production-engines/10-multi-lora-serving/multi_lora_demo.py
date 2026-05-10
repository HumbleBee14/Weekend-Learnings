"""
10 - Multi-LoRA serving demo against vLLM.

Prereqs:
    # Train two tiny LoRAs (separate scripts; 5-10 min each on a small GPU).
    # Code LoRA: SFT on a few thousand code-style instructions.
    # Poetry LoRA: SFT on a poetry instruction set.
    # Save each to ./adapters/code and ./adapters/poetry in PEFT format.

Serve:
    vllm serve Qwen/Qwen2.5-7B-Instruct \
        --enable-lora \
        --max-loras 4 \
        --max-lora-rank 64 \
        --lora-modules code=./adapters/code poetry=./adapters/poetry

Run:
    pip install openai
    python multi_lora_demo.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from openai import AsyncOpenAI


PROMPTS_CODE = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Refactor this function to be tail-recursive: def f(n): return 1 if n<=1 else n*f(n-1)",
    "What's wrong with `for k, v in d.iteritems()` in Python 3?",
]

PROMPTS_POETRY = [
    "A four-line poem about paged KV cache.",
    "Haiku for a debugger watching a TTFT spike.",
    "A short verse about rented GPUs at 3am.",
]


async def hit(client: AsyncOpenAI, model: str, prompt: str) -> tuple[float, int, float]:
    t0 = time.perf_counter()
    ttft: float | None = None
    n = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        stream=True,
        temperature=0.7,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
    return ttft or 0.0, n, time.perf_counter() - t0


async def run_one_adapter(client: AsyncOpenAI, model: str, prompts: list[str], label: str) -> None:
    t0 = time.perf_counter()
    out = await asyncio.gather(*[hit(client, model, p) for p in prompts])
    wall = time.perf_counter() - t0
    ttfts = [r[0] * 1000 for r in out]
    toks = sum(r[1] for r in out)
    print(f"  [{label:20s}]  {toks} tok in {wall:.2f}s = {toks/wall:.0f} tok/s   "
          f"TTFT mean {statistics.mean(ttfts):.0f} ms")


async def run_interleaved(client: AsyncOpenAI) -> None:
    pairs = list(zip(PROMPTS_CODE, PROMPTS_POETRY))
    tasks: list = []
    t0 = time.perf_counter()
    for code_p, poetry_p in pairs:
        tasks.append(hit(client, "code", code_p))
        tasks.append(hit(client, "poetry", poetry_p))
    out = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    toks = sum(r[1] for r in out)
    ttfts = [r[0] * 1000 for r in out]
    print(f"  [interleaved        ]  {toks} tok in {wall:.2f}s = {toks/wall:.0f} tok/s   "
          f"TTFT mean {statistics.mean(ttfts):.0f} ms")


async def main() -> None:
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    print("\nSingle-adapter baselines:")
    await run_one_adapter(client, "Qwen/Qwen2.5-7B-Instruct", PROMPTS_CODE + PROMPTS_POETRY, "base (no LoRA)")
    await run_one_adapter(client, "code", PROMPTS_CODE, "code LoRA")
    await run_one_adapter(client, "poetry", PROMPTS_POETRY, "poetry LoRA")

    print("\nMixed batch (this is the multi-LoRA test):")
    await run_interleaved(client)

    print("\nThe interleaved row should be within ~10-20% of the no-LoRA baseline.")
    print("If it isn't, --max-loras is probably too low and adapters are thrashing.")


if __name__ == "__main__":
    asyncio.run(main())
