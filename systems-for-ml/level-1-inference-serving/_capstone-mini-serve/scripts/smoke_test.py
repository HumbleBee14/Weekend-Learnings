"""End-to-end smoke test: start server, hit health, send blocking + streaming requests."""

import asyncio
import json

import httpx

URL = "http://localhost:8000"


async def main():
    async with httpx.AsyncClient() as client:
        # 1. Health check
        r = await client.get(f"{URL}/health")
        print("health:", r.json())

        # 2. Blocking generate
        r = await client.post(
            f"{URL}/generate",
            json={"prompt": "Say hi in 5 words.", "max_tokens": 20},
            timeout=120,
        )
        body = r.json()
        print("blocking:", body["completion"])
        print("metrics:", body["metrics"])

        # 3. Streaming generate
        print("streaming:", end=" ", flush=True)
        async with client.stream(
            "POST",
            f"{URL}/generate_stream",
            json={"prompt": "Count to five slowly.", "max_tokens": 30},
            timeout=120,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if "token" in data:
                    print(data["token"], end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
