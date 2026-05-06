# 01 — Inference Server Basics

## The setup

An inference server has one job: receive a prompt over HTTP, run it through a model, return the completion. The naive version is 30 lines. Production versions are 30,000. The next 8 weeks are about that gap.

This topic builds the 30-line version. It works. It's also wrong in interesting ways — and those wrong ways are exactly what every later optimization fixes.

## What `model.generate()` actually does

`model.generate(input_ids, max_new_tokens=100)` runs an autoregressive loop:

```
1. Run the prompt through the model → logits for token 101
2. Sample a token from those logits (greedy / top-k / etc.) → token 101
3. Append token 101 to the input → input is now 101 tokens
4. Run the new input through the model → logits for token 102
5. Sample → token 102
6. Repeat until max_new_tokens or EOS
```

There are two distinct phases here:

- **Prefill** — step 1. The whole prompt is processed in one forward pass. Compute-heavy: it's a large matmul over many tokens at once.
- **Decode** — steps 2 through end. Each step processes exactly one new token. Memory-bandwidth-heavy: for each token you read the entire model's weights from HBM but do very little compute per byte read.

Prefill is compute-bound. Decode is memory-bound. **This single fact is the most important thing in LLM serving.** Every modern engine (vLLM, SGLang, TRT-LLM) treats them as separate problems — chunked prefill, disaggregated serving, paged KV cache all exist because of this split.

## Why HTTP

You could call a Python function directly. HTTP gives you:

- Decoupling — the model runs on a GPU machine, clients can be anywhere
- Standardization — OpenAI's API is HTTP+JSON, every chat UI speaks this
- Operational maturity — load balancers, observability, rate limiters all speak HTTP

FastAPI is the standard Python framework for this. Async-native, JSON validation via Pydantic.

## Why one worker, not many

`uvicorn main:app --workers 4` runs four separate Python processes, each with its own copy of the model. At low load this looks like batching is working — but it's really four independent models on the same GPU fighting for memory and compute.

Stay on `--workers 1` so the batching results in topic 03 are honest.

## The GIL trap

Python has a Global Interpreter Lock. Only one Python thread runs Python code at a time.

When request 1 is inside `model.generate()`, request 2 waits — even though FastAPI is async.

Why doesn't `await` help? Because `model.generate()` is a long synchronous CUDA call. Python's `await` only releases the event loop on I/O, not on CPU/GPU compute. The two requests serialize on the server even though they overlap on the wire.

This is the entire reason batching exists. Instead of serializing two requests, you stack them into one model call.

## The three measurements that matter

1. **Single-request latency** — baseline, one request end-to-end
2. **Sequential 10-request latency** — 10 requests one after another, ≈ 10× single
3. **Concurrent 10-request latency** — 10 requests at the same time via `asyncio.gather`

The point: (3) ≈ (2). Concurrent requests do not actually run concurrently on a single-worker server. That gap is the motivation for the rest of the curriculum.

## Pitfalls

1. **Picking a 7B+ model.** Use 0.5B–1.5B for the whole week. Iteration speed beats model size for learning systems lessons.
2. **Multi-worker uvicorn.** 1 worker, period.
3. **`curl` in a shell loop.** That's sequential. Use `asyncio.gather` from a Python script.
4. **No warmup.** First request includes CUDA context init + kernel JIT. Always do at least one warmup before timing.

## Task

Read `server.py` line by line. Run it. Run `measure.py`. Confirm the GIL serialization story in your own data.
