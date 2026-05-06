# 02 — Streaming Tokens

## Files

- `CONCEPTS.md` — SSE format, TTFT vs ITL, why we need a background thread for the streamer
- `server.py` — adds `/generate_stream` endpoint to the topic-01 server
- `measure.py` — opens an SSE stream and reports TTFT + ITL

## Quickstart

```bash
pip install fastapi uvicorn transformers torch httpx
uvicorn server:app --workers 1 --port 8000

# In another terminal:
curl -N -X POST http://localhost:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Tell me a short joke.", "max_tokens":50}'

# Or run the measurement script:
python measure.py
```

`-N` on curl disables buffering — without it you wait for the whole response.

## Expected output

```json
{
  "server_ttft_ms": 180,    // prefill time
  "client_ttft_ms": 195,    // what the user feels (includes network)
  "itl_median_ms": 45,      // gap between tokens
  "tokens": 100,
  "total_ms": 4680
}
```

The user sees the first token at ~200ms even though the full response takes 4700ms. That perception gap is the win.

## Try

- **Increase prompt length to 1000 tokens.** TTFT goes up linearly (prefill scales with prompt length). ITL stays roughly constant.
- **Open the stream in a browser via `<eventsource>` or fetch a stream from JS.** It Just Works — that's the SSE story.
- **Send the same request via the topic-01 `/generate` endpoint.** Compare total latency. Same throughput, very different UX.
