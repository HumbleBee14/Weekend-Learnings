# 02 — Streaming Tokens

## The point

In topic 01, the user waits 5 seconds for the model to finish, then gets the whole response at once. That feels slow even when throughput is fine.

Stream tokens as they're generated and the user sees the first word in 200ms — even if total time is still 5 seconds. **Throughput didn't change. Perceived latency dropped 25×.** This is the single most important UX insight in LLM serving.

## SSE vs the alternatives

Three options for "server pushes data over HTTP":

| Protocol | When |
|---|---|
| Polling (`GET /status` repeatedly) | Wrong for token streams — too slow, too chatty |
| WebSockets | Bidirectional real-time (multiplayer games, collaborative editing) |
| Server-Sent Events (SSE) | One-way server → client streams |

Tokens flow one direction (server → client) and need to start *immediately*. SSE is the right tool. Plain HTTP — works through proxies, load balancers, browsers, CDNs. No special infra.

## SSE wire format

```
data: {"token": "Hello"}

data: {"token": " world"}

data: [DONE]

```

Each event ends with `\n\n` (double newline). Content-Type is `text/event-stream`. That's it.

## TTFT vs ITL vs total latency

- **TTFT (Time To First Token)** — request received → first token leaves. Dominated by *prefill* time. The metric users feel as "responsiveness."
- **ITL (Inter-Token Latency)** — average gap between consecutive tokens during decode. The metric users feel as "speed of typing."
- **Total latency** — TTFT + (output_tokens × ITL). What topic 01 measured.

100ms TTFT with 50ms ITL feels great. 1000ms TTFT with 20ms ITL feels sluggish even at similar total time. **Optimize TTFT first, then ITL — almost always in that order.**

## How streaming works inside HuggingFace

`model.generate()` normally runs the full loop and returns the final tensor. To stream, you need a callback that fires per token.

`TextIteratorStreamer` is HuggingFace's queue-based solution. Run `model.generate(streamer=streamer)` in a background thread; the main thread reads tokens off the queue as they arrive.

```python
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
thread = Thread(target=model.generate, kwargs={..., "streamer": streamer})
thread.start()
for new_text in streamer:
    yield new_text
```

The thread runs `generate()`. We read the queue. Without the thread, the streamer iterator would block — `generate()` and the consumer would deadlock.

## FastAPI's StreamingResponse

```python
return StreamingResponse(
    sse_generator(),
    media_type="text/event-stream",
)
```

The response stays open until the generator finishes. Anything yielded gets pushed to the client immediately.

## Pitfalls

1. **Single `\n` instead of `\n\n`.** Client buffers forever waiting for the event to end.
2. **String-concat tokens with quotes/newlines.** Use `json.dumps()` per chunk.
3. **Testing with `requests`.** It buffers the full response. Use `httpx` with `stream=True`, or `curl -N`.
4. **Async generator that's secretly sync.** `yield from` a sync iterator blocks the event loop. For one server it's fine; at scale, run the streamer in a thread pool via `asyncio.to_thread`.
