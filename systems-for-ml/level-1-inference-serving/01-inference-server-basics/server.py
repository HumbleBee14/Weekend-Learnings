"""
The simplest possible LLM inference server.

What's here:
  - One FastAPI route: POST /generate
  - Loads a small HuggingFace model into memory at startup
  - Calls model.generate() synchronously and returns the result

What's intentionally missing (and we'll add over the week):
  - Streaming (you wait for the full completion before getting any tokens)
  - Batching (each request gets its own forward pass)
  - Paged KV cache (memory grows linearly with context length)
  - Continuous batching (a slow request blocks fast ones from joining)

Run it:
    pip install fastapi uvicorn transformers torch
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

from contextlib import asynccontextmanager
from time import perf_counter

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pick a small model. 0.5B fits on CPU; iteration speed matters more than model size for learning.
# Swap to "Qwen/Qwen2.5-1.5B-Instruct" if you have a GPU and want slightly better outputs.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Module-level state. Populated in lifespan (below) so it's loaded once at server startup.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler — runs at startup and shutdown.

    Why this matters: model loading is slow (seconds for tiny models, minutes for large ones).
    You do NOT want to load the model on every request. Load once, hold in memory, serve forever.
    """
    print(f"Loading {MODEL_ID}...")
    t0 = perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",  # GPU if available, else CPU
    )
    model.eval()  # turn off dropout etc

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["device"] = next(model.parameters()).device

    print(f"Loaded in {perf_counter() - t0:.2f}s on {state['device']}")
    yield  # server runs

    # Shutdown — free GPU memory if we're on CUDA
    state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    """Request schema — Pydantic validates incoming JSON automatically."""

    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    completion: str
    tokens_generated: int
    latency_ms: float


@app.get("/health")
def health():
    """Liveness probe — useful for load balancers and Kubernetes."""
    return {"status": "ok", "model": MODEL_ID, "device": str(state.get("device"))}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    The hot path.

    NOTE: this is intentionally a synchronous function (not `async def`).
    `model.generate()` is a blocking GPU call — there's nothing to `await` on.
    Marking it `async def` would *look* concurrent but would still serialize on the GIL.
    """
    t0 = perf_counter()

    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]

    # Apply the chat template — modern instruction-tuned models expect this format.
    # If you skip this, the model still runs but quality drops noticeably.
    messages = [{"role": "user", "content": req.prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and move to device. `return_tensors="pt"` gives us PyTorch tensors.
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Generate. `do_sample=True` + temperature > 0 enables sampling (else greedy).
    # `pad_token_id` shuts up a warning when the tokenizer has no explicit pad token.
    with torch.inference_mode():  # like no_grad but slightly faster
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens — we only want the new completion.
    new_token_ids = output_ids[0, input_len:]
    completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    latency_ms = (perf_counter() - t0) * 1000.0

    return GenerateResponse(
        completion=completion,
        tokens_generated=len(new_token_ids),
        latency_ms=latency_ms,
    )
