"""
14 - VLM serving demo against vLLM.

Sends image+text requests to a vLLM server running Qwen2.5-VL (or any
OpenAI-compatible VLM endpoint) and measures the three relevant phases:
TTFT (which includes vision encode + LLM prefill), output throughput,
and prefix-cache effectiveness across same-image batches.

Prereqs:
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 \
        --max-model-len 8192 --limit-mm-per-prompt image=2

    pip install openai pillow

Run:
    python vlm_demo.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

try:
    from PIL import Image  # type: ignore
except ImportError:
    print("pip install pillow")
    raise SystemExit(1)


def synthetic_image(size: int, color: tuple[int, int, int]) -> str:
    """Return a base64-encoded PNG of size x size in the given color."""
    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def hit(client: AsyncOpenAI, model: str, image_b64: str, text: str, max_tokens: int) -> tuple[float, int, float]:
    t0 = time.perf_counter()
    ttft: float | None = None
    n = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": text},
            ],
        }],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
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


async def scenario_same_image_varied_text(client: AsyncOpenAI, model: str) -> None:
    print("\n[A] Same image, varied text — visual prefix-cache should kick in after #1")
    img = synthetic_image(448, (180, 60, 60))
    questions = [
        "What primary color is the image?",
        "Describe the image in five words.",
        "Could this be a sunset?",
        "What does this look like?",
        "Is the image uniform?",
    ]
    ttfts: list[float] = []
    for q in questions:
        ttft, _, total = await hit(client, model, img, q, 64)
        print(f"   ttft={ttft*1000:.0f}ms  total={total*1000:.0f}ms  q={q}")
        ttfts.append(ttft)
    print(f"   first vs subsequent TTFT: {ttfts[0]*1000:.0f} vs mean(rest) {statistics.mean(ttfts[1:])*1000:.0f} ms")


async def scenario_different_images(client: AsyncOpenAI, model: str) -> None:
    print("\n[B] Different images, same text — must NOT false-share cache")
    images = [
        synthetic_image(448, (200, 50, 50)),
        synthetic_image(448, (50, 200, 50)),
        synthetic_image(448, (50, 50, 200)),
        synthetic_image(448, (200, 200, 50)),
    ]
    text = "What primary color is the image?"
    for i, img in enumerate(images):
        _, _, total = await hit(client, model, img, text, 32)
        print(f"   image #{i}  total={total*1000:.0f}ms  text='{text}'")
    print("   answers should differ per color — if they don't, image hashing is broken")


async def scenario_size_sweep(client: AsyncOpenAI, model: str) -> None:
    print("\n[C] Image size sweep — visual token count varies dramatically")
    sizes = [256, 448, 672, 1024]
    for s in sizes:
        img = synthetic_image(s, (100, 150, 200))
        ttft, _, total = await hit(client, model, img, "Describe this image briefly.", 64)
        print(f"   {s}x{s}px  ttft={ttft*1000:.0f}ms  total={total*1000:.0f}ms")
    print("   TTFT should grow super-linearly with image size — visual encoder + prefill cost.")


async def main() -> None:
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    model = "Qwen/Qwen2.5-VL-7B-Instruct"
    await scenario_same_image_varied_text(client, model)
    await scenario_different_images(client, model)
    await scenario_size_sweep(client, model)


if __name__ == "__main__":
    asyncio.run(main())
