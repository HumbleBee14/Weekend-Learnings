"""
Server with naive static batching.

A background asyncio task drains an internal queue every `MAX_WAIT_MS` (or as soon as the
queue has `MAX_BATCH_SIZE` items), batches the requests, runs ONE forward pass for the
whole batch, and resolves each request's future with its slice of the output.

This is the simplest "batching" — purely static, no continuous batching, no paged KV.
The padding and head-of-line blocking pain points are intentional: feel them now,
fix them in Level 4.

Run:
    uvicorn server:app --workers 1 --port 8000
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from time import perf_counter

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_BATCH_SIZE = 8
MAX_WAIT_MS = 10  # how long the batcher waits before launching a partial batch

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID)
    state["model"] = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    state["model"].eval()
    state["device"] = next(state["model"].parameters()).device

    # Some models (LLaMA family) have no pad_token; reuse eos_token for padding
    if state["tokenizer"].pad_token is None:
        state["tokenizer"].pad_token = state["tokenizer"].eos_token

    state["queue"] = asyncio.Queue()
    state["batcher_task"] = asyncio.create_task(batcher_loop())

    print(f"Ready on {state['device']} | batch_size={MAX_BATCH_SIZE} | max_wait={MAX_WAIT_MS}ms")
    yield

    state["batcher_task"].cancel()
    state.clear()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7


class QueuedRequest:
    """One request waiting in the batcher queue. The future is resolved when generation finishes."""

    def __init__(self, prompt: str, max_tokens: int, temperature: float):
        self.id = uuid.uuid4().hex[:8]
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.enqueue_time = perf_counter()


async def batcher_loop():
    """
    Runs forever. Each iteration:
      1. Wait for at least one request to arrive (block on queue.get())
      2. Greedily drain up to MAX_BATCH_SIZE more, or until MAX_WAIT_MS elapsed
      3. Run model.generate() on the whole batch
      4. Resolve each request's future with its slice of the output
    """
    while True:
        try:
            batch: list[QueuedRequest] = []
            # Block until first request arrives
            first = await state["queue"].get()
            batch.append(first)

            # Try to fill the batch within the wait budget
            deadline = perf_counter() + MAX_WAIT_MS / 1000.0
            while len(batch) < MAX_BATCH_SIZE:
                remaining_ms = (deadline - perf_counter()) * 1000.0
                if remaining_ms <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(state["queue"].get(), timeout=remaining_ms / 1000.0)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break

            await run_batch(batch)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"Batcher error: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


async def run_batch(batch: list[QueuedRequest]) -> None:
    """Execute one forward pass for the whole batch. Resolve each future."""
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]

    t_run_start = perf_counter()

    # Apply chat template per request, then tokenize the batch with padding
    prompts = []
    for req in batch:
        messages = [{"role": "user", "content": req.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,           # left-pads (default) so all sequences end at the same place
        padding_side="left",    # critical for generation: new tokens append to the right
    ).to(device)

    input_lengths = inputs["attention_mask"].sum(dim=1)  # actual length per row

    # Use the largest max_tokens in the batch — others will hit EOS or hit this cap
    max_new = max(req.max_tokens for req in batch)
    # Use the highest temperature in the batch (or 0 if all want greedy). In real systems
    # you'd separate by sampling params. For learning we keep it simple.
    temperature = max(req.temperature for req in batch)

    # ONE forward pass for the whole batch
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the prompt tokens per row, decode, resolve futures
    prompt_len = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[:, prompt_len:]

    batch_run_ms = (perf_counter() - t_run_start) * 1000

    for i, req in enumerate(batch):
        gen_ids = new_token_ids[i]
        # Trim trailing padding / EOS
        eos_positions = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_ids = gen_ids[: eos_positions[0]]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

        wait_ms = (t_run_start - req.enqueue_time) * 1000
        req.future.set_result({
            "completion": completion,
            "tokens_generated": len(gen_ids),
            "batch_size": len(batch),
            "queue_wait_ms": wait_ms,
            "batch_run_ms": batch_run_ms,
        })


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    The endpoint is now async — it submits to the queue and awaits the future.
    The actual model.generate() runs in the batcher loop.
    """
    qr = QueuedRequest(req.prompt, req.max_tokens, req.temperature)
    await state["queue"].put(qr)
    return await qr.future
