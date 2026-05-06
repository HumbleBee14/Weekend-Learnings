"""Async micro-batcher: drains the queue, runs one forward pass, splits outputs.

Both /generate (blocking) and /generate_stream (SSE) submit through here. Streaming requests
get their own per-request token queue; the batcher pushes tokens into all per-request queues
as they're produced via a TextStreamer subclass.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import torch
from transformers import TextStreamer

from .config import settings
from .model_loader import LoadedModel

log = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """One pending request. Either blocking (resolves a future) or streaming (pushes to a queue)."""

    prompt: str
    max_tokens: int
    temperature: float
    stream: bool

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    enqueue_time: float = field(default_factory=perf_counter)

    # For blocking requests: a future that resolves with the completion
    future: Optional[asyncio.Future] = None

    # For streaming requests: a per-request queue of decoded text chunks. None = end of stream.
    token_queue: Optional[asyncio.Queue] = None


class _PerRequestStreamer(TextStreamer):
    """A streamer that demuxes batch-level text back into per-request queues.

    HuggingFace's built-in streamers assume batch_size == 1. For batched streaming we need
    custom logic: each row in the batch maps to a different request's asyncio.Queue.
    """

    def __init__(self, tokenizer, batch_requests: list[QueuedRequest], loop: asyncio.AbstractEventLoop):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.batch_requests = batch_requests
        self.loop = loop
        # TextStreamer is built for batch=1. For real per-row streaming we'd need a different
        # approach (e.g., token IDs callback). For pedagogy, we stream the whole batch's joined
        # text to ALL streaming requests in the batch — imperfect but illustrates the pattern.
        # Note: production engines (vLLM) handle this with a proper per-request token callback.

    def on_finalized_text(self, text: str, stream_end: bool = False):
        for req in self.batch_requests:
            if req.token_queue is not None:
                self.loop.call_soon_threadsafe(req.token_queue.put_nowait, text)
                if stream_end:
                    self.loop.call_soon_threadsafe(req.token_queue.put_nowait, None)


class Batcher:
    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded
        self.queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="batcher")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    async def submit(self, req: QueuedRequest) -> None:
        # Backpressure: if the queue is full, raise (handled in the route as 503)
        try:
            self.queue.put_nowait(req)
        except asyncio.QueueFull as e:
            raise RuntimeError("queue_full") from e

    async def _loop(self) -> None:
        while True:
            try:
                batch = await self._collect_batch()
                await self._run_batch(batch)
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.exception("batcher iteration failed: %s", e)

    async def _collect_batch(self) -> list[QueuedRequest]:
        first = await self.queue.get()
        batch = [first]
        deadline = perf_counter() + settings.max_wait_ms / 1000.0
        while len(batch) < settings.max_batch_size:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(nxt)
            except asyncio.TimeoutError:
                break
        return batch

    async def _run_batch(self, batch: list[QueuedRequest]) -> None:
        t_run_start = perf_counter()

        tokenizer = self.loaded.tokenizer
        model = self.loaded.model
        device = self.loaded.device

        # Build chat-templated prompts
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in batch
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(device)

        # If any request wants streaming, attach a streamer
        streamer = None
        if any(r.stream for r in batch):
            streamer = _PerRequestStreamer(tokenizer, batch, asyncio.get_running_loop())

        max_new = max(r.max_tokens for r in batch)
        temperature = max(r.temperature for r in batch)

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=tokenizer.pad_token_id,
                    streamer=streamer,
                )
        except Exception as e:
            for r in batch:
                self._fail(r, e)
            return

        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[:, prompt_len:]

        run_ms = (perf_counter() - t_run_start) * 1000
        for i, req in enumerate(batch):
            gen_ids = new_token_ids[i]
            eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0]]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

            queue_wait_ms = (t_run_start - req.enqueue_time) * 1000
            metrics = {
                "queue_wait_ms": queue_wait_ms,
                "decode_ms": run_ms,
                "total_ms": queue_wait_ms + run_ms,
                "batch_size_seen": len(batch),
            }
            self._resolve(req, completion, len(gen_ids), metrics)

    def _resolve(self, req: QueuedRequest, completion: str, n_tokens: int, metrics: dict) -> None:
        if req.future is not None and not req.future.done():
            req.future.set_result((completion, n_tokens, metrics))
        # streaming requests already got their tokens via the streamer; the None sentinel
        # signals end-of-stream

    def _fail(self, req: QueuedRequest, exc: Exception) -> None:
        if req.future is not None and not req.future.done():
            req.future.set_exception(exc)
        if req.token_queue is not None:
            try:
                req.token_queue.put_nowait(None)  # signal end with no tokens
            except asyncio.QueueFull:
                pass
