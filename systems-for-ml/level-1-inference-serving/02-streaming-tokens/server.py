"""
Streaming inference server using Server-Sent Events.

Compared to topic 01's server, the changes are:
  - New endpoint: POST /generate_stream
  - Uses TextIteratorStreamer + a background thread for generation
  - Yields SSE-formatted chunks as tokens are generated

Run it:
    uvicorn server:app --workers 1 --port 8000
"""

import json
from contextlib import asynccontextmanager
from threading import Thread
from time import perf_counter

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_ID}...")
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID)
    state["model"] = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    state["model"].eval()
    state["device"] = next(state["model"].parameters()).device
    print(f"Ready on {state['device']}")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):
    """
    Stream tokens via SSE. Each line is `data: {"token": "..."}\\n\\n`.

    Why not async def: model.generate() runs in a thread we spawn ourselves; the FastAPI
    handler iterates the streamer queue. Mixing async + threads is doable but adds confusion
    without a real benefit here.
    """
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]

    messages = [{"role": "user", "content": req.prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # The streamer is a queue. As model.generate produces tokens, it pushes them here.
    # skip_prompt=True means we don't echo the input back.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # Run generation in a background thread; the main coroutine consumes the stream.
    # If we did not use a thread, the streamer iterator would block indefinitely.
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def event_stream():
        """Generator that emits SSE events as tokens arrive."""
        request_start = perf_counter()
        first_token_time = None
        token_count = 0

        # Tell the client we're ready. Useful for measuring TTFT vs request-handling overhead.
        yield f"data: {json.dumps({'event': 'start'})}\n\n"

        for new_text in streamer:
            if first_token_time is None:
                first_token_time = perf_counter()
                ttft_ms = (first_token_time - request_start) * 1000.0
                yield f"data: {json.dumps({'event': 'first_token', 'ttft_ms': ttft_ms})}\n\n"

            token_count += 1
            yield f"data: {json.dumps({'token': new_text})}\n\n"

        thread.join()
        total_ms = (perf_counter() - request_start) * 1000.0
        yield f"data: {json.dumps({'event': 'done', 'tokens': token_count, 'total_ms': total_ms})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
