"""FastAPI app — wires routes to the batcher.

What's deliberately structured this way:
  - lifespan loads the model once and starts the batcher
  - dependencies (model, batcher) injected through app.state
  - routes are thin: validate, submit, return
  - errors map to proper HTTP status codes
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import model_loader
from .batcher import Batcher, QueuedRequest
from .config import settings
from .schemas import GenerateRequest, GenerateResponse, HealthResponse, RequestMetrics

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("mini_serve")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting mini-serve")
    loaded = model_loader.load()
    batcher = Batcher(loaded)
    await batcher.start()
    app.state.loaded = loaded
    app.state.batcher = batcher
    log.info(
        "ready: model=%s device=%s batch_size=%d max_wait_ms=%d",
        settings.model_id, loaded.device, settings.max_batch_size, settings.max_wait_ms,
    )
    yield
    log.info("shutting down")
    await batcher.stop()


app = FastAPI(title="mini-serve", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    batcher: Batcher = request.app.state.batcher
    loaded = request.app.state.loaded
    return HealthResponse(
        status="ok",
        model=settings.model_id,
        device=str(loaded.device),
        queue_depth=batcher.queue_depth,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    batcher: Batcher = request.app.state.batcher
    qr = QueuedRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=False,
        future=asyncio.get_event_loop().create_future(),
    )
    try:
        await batcher.submit(qr)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="server overloaded; retry later")

    completion, n_tokens, metrics = await qr.future
    return GenerateResponse(
        completion=completion,
        tokens_generated=n_tokens,
        request_id=qr.request_id,
        metrics=RequestMetrics(**metrics),
    )


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest, request: Request):
    batcher: Batcher = request.app.state.batcher
    qr = QueuedRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=True,
        token_queue=asyncio.Queue(),
    )
    try:
        await batcher.submit(qr)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="server overloaded; retry later")

    async def event_stream():
        yield f"data: {json.dumps({'event': 'start', 'request_id': qr.request_id})}\n\n"
        while True:
            chunk = await qr.token_queue.get()
            if chunk is None:
                break
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
