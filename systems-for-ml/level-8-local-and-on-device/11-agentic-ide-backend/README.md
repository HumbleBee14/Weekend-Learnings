# 11 — Agentic IDE Backend

## Files

- `CONCEPTS.md` — what changes when the model runs locally, the sub-100 ms TTFT budget, two-model architecture, the agent loop, cloud-vs-local honest comparison.
- `agent_loop.py` — minimal multi-tool agent (file read, file edit via patch, shell exec) hitting any OpenAI-compatible endpoint. Streams. Step-limited.

## Quickstart

```bash
# Have an OpenAI-compatible local server running first (Topic 08).
# e.g. ollama serve (with OLLAMA_BACKEND=mlx) and ollama pull qwen2.5-coder:7b

pip install openai
python agent_loop.py \
    --base-url http://localhost:11434/v1 \
    --model qwen2.5-coder:7b \
    --task "Read README.md, then write a one-line summary to summary.txt"
```

## Expected output

```
[step 1] tool=read_file path=README.md
[step 2] tool=run_shell cmd='echo "..." > summary.txt'
[done] task complete in 2 steps, 1.4s wall, 0 retries
```

## Try

- Add a `--ttft` flag and time the first token. Hit it with both `qwen2.5-coder:1.5b` (fast) and `qwen2.5-coder:7b` (smart). Note the TTFT gap — this is why you use two models.
- Force a malformed-tool-call situation: prompt the model with a confusing system message. Without `response_format=json_schema`, watch the parse fail. Then re-run with vLLM-MLX and `response_format` enabled and see it pass every time.
- Compare: run the same task against `https://api.anthropic.com` (with adapter) and capture wall time + cost. **G18/G19/G20** pieces.
- Drop `qwen3-coder:32b` (MoE) in for the chat model and re-run a multi-step task. Quality up, throughput similar to 7B dense thanks to MoE active-bandwidth math (Topic 09).

## Where this goes

This is the runtime for **Project 4 (`local-agent`)**. Topic 12 fine-tunes a model into your code style; Topic 13 layers preference learning. Topic 16 closes with the privacy threat model that lets you actually claim "private agent" honestly.
