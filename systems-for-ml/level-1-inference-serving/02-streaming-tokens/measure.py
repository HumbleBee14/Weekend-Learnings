"""
Measure TTFT and ITL on the streaming endpoint.

Run:
    1. Terminal A: uvicorn server:app --workers 1
    2. Terminal B: python measure.py
"""

import asyncio
import json
import statistics
from time import perf_counter

import httpx

URL = "http://localhost:8000/generate_stream"
PROMPT = "Explain how a hash table works in three paragraphs."


async def stream_one(client: httpx.AsyncClient) -> dict:
    """Open an SSE stream, time TTFT and per-token ITL."""
    timestamps = []  # wall-clock time of every token chunk
    request_start = perf_counter()
    server_ttft = None
    total_tokens = 0

    async with client.stream("POST", URL, json={"prompt": PROMPT, "max_tokens": 100}, timeout=120) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if "ttft_ms" in data:
                # Server-reported TTFT (request → first generate step)
                server_ttft = data["ttft_ms"]
            elif "token" in data:
                timestamps.append(perf_counter())
                total_tokens += 1

    if not timestamps:
        return {"error": "no tokens received"}

    client_ttft_ms = (timestamps[0] - request_start) * 1000
    inter_token_gaps_ms = [
        (timestamps[i] - timestamps[i - 1]) * 1000 for i in range(1, len(timestamps))
    ]
    return {
        "server_ttft_ms": server_ttft,
        "client_ttft_ms": client_ttft_ms,
        "itl_median_ms": statistics.median(inter_token_gaps_ms) if inter_token_gaps_ms else None,
        "itl_p95_ms": statistics.quantiles(inter_token_gaps_ms, n=20)[18] if len(inter_token_gaps_ms) >= 20 else None,
        "tokens": total_tokens,
        "total_ms": (timestamps[-1] - request_start) * 1000,
    }


async def main():
    async with httpx.AsyncClient() as client:
        # Warmup
        print("Warmup...")
        await stream_one(client)

        # Single stream
        print("\nSingle streaming request...")
        result = await stream_one(client)
        print(json.dumps(result, indent=2))

        print("\n--- Notes ---")
        print("server_ttft_ms = time from request hitting handler to first generate token (prefill cost).")
        print("client_ttft_ms = above + network + framework overhead (the user-felt number).")
        print(f"itl_median_ms  = the 'speed of typing' the user perceives.")
        print(f"For a 0.5B model on CPU, expect TTFT ≈ 100-500ms, ITL ≈ 30-80ms.")


if __name__ == "__main__":
    asyncio.run(main())
