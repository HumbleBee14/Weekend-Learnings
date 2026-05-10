"""
Cancellation-propagating router fragment for reasoning-aware serving.

Wraps the upstream LLM stream so that:
  - Client disconnect -> upstream connection closed -> engine frees decode slot.
  - The propagation chain is traceable end-to-end.

Use as a reference implementation for the Topic 06 router's streaming handler.

Demo (no real engine):
    python cancellation_router.py
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn


# ---------- the load-bearing pattern ----------

@asynccontextmanager
async def upstream_stream(client: httpx.AsyncClient, url: str, body: dict):
    """
    Open an upstream SSE stream and yield its iterator. On exit (incl. cancel),
    the underlying connection is closed; vLLM's scheduler observes the EOF and
    frees the decode slot.
    """
    async with client.stream("POST", url, json=body) as upstream:
        yield upstream


async def proxy_stream(request: Request, upstream_url: str, body: dict):
    """
    Yield bytes from upstream while polling client liveness. On disconnect,
    cancel the upstream coroutine — which closes the connection and frees the
    engine's decode slot.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        async with upstream_stream(client, upstream_url, body) as up:
            async for chunk in up.aiter_raw():
                if await request.is_disconnected():
                    # Closes the `async with client.stream(...)` cleanly.
                    return
                yield chunk


# ---------- standalone demo: a stub upstream that decodes slowly ----------

stub = FastAPI()
DECODE_SECONDS = 30


@stub.post("/v1/completions")
async def stub_decode():
    async def gen():
        try:
            for i in range(DECODE_SECONDS * 5):
                await asyncio.sleep(0.2)
                yield f"data: token-{i}\n\n".encode()
        except asyncio.CancelledError:
            print(f"[stub] decode cancelled at token {i}, freeing slot.")
            raise

    return StreamingResponse(gen(), media_type="text/event-stream")


router_app = FastAPI()


@router_app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    return StreamingResponse(
        proxy_stream(request, "http://localhost:8001/v1/completions", body),
        media_type="text/event-stream",
    )


def main():
    import multiprocessing as mp

    def run_stub():
        uvicorn.run(stub, host="0.0.0.0", port=8001, log_level="warning")

    p = mp.Process(target=run_stub, daemon=True)
    p.start()
    time.sleep(0.5)
    print("Stub upstream running on :8001")
    print("Router running on :8080")
    print("Disconnect mid-stream and watch the stub print the cancel.")
    uvicorn.run(router_app, host="0.0.0.0", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
