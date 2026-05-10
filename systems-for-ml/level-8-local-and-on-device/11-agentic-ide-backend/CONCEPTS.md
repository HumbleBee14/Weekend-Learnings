# 11 — Agentic IDE Backend

## What changes when the model runs locally

A cloud-API agentic loop pays for two things on every tool call:

1. Network round-trip (200–500 ms US <-> US-East, 600–800 ms transcontinental).
2. Per-token cost ($0.003–$0.030 per 1k tokens at 2026 prices on big models).

A local agentic loop has neither. The interesting consequences:

- **Sub-100 ms TTFT is achievable** for small-prompt completions — well below the 250+ ms floor of any cloud API. That changes the *interaction grammar*: completions feel like predictive text rather than network chat.
- **Free retries.** When the agent screws up step 4, you can re-plan from scratch instead of treating tokens as expensive.
- **Run a smaller model 100x more often** rather than a bigger model carefully. The economics flip.
- **Privacy.** Code, prompts, RAG corpus stay on-device (Topic 16).

## The latency budget for "feels instant"

Human perception lands at ~100 ms for "instant" responses. Sub-50 ms is imperceptible. The cloud cannot offer this — physics. Local can:

```
  +--------------------------------+
  |  100 ms total budget           |
  +--------------------------------+
   keystroke -> request build-up   ~  5 ms
   tokenize input                  ~  2 ms
   prefill (300-token prompt)      ~ 30 ms  (Qwen2.5-Coder 1.5B at 9000 tps)
   first decoded token             ~ 15 ms
   network (loopback)              ~  1 ms
   render in editor                ~  5 ms
                                   --------
                                   ~ 60 ms TTFT, 40 ms slack
```

The dominant cost is prefill. Two implications:

- Use a small model (1.5B–3B) for completion. A 7B will not fit the budget.
- Cache aggressively. Common prefixes (the first N lines of the file) hit a cached KV (Level 4 Topic 11) and skip prefill entirely.

## Two-model architecture

```
                  +--------------------------------+
                  |  Editor (Cursor / Zed / TUI)   |
                  +--------------------------------+
                              |   |
            autocomplete      |   |   chat / agent
            (high frequency,  |   |   (low frequency,
             tight latency)   |   |    longer reasoning)
                              v   v
       +-----------------------------+    +-----------------------------+
       |  Coder small (1.5-3B)       |    |  Coder large (7-32B)        |
       |  served by mlx_lm.server    |    |  served by Ollama-MLX       |
       |  prefix-cache aggressive    |    |  tool-calling JSON-strict   |
       |  no spec decode (overhead)  |    |  spec decode on             |
       +-----------------------------+    +-----------------------------+
                              \                 /
                               \               /
                                v             v
                        +---------------------------+
                        |  Shared embedding model   |
                        |  bge-m3 / nomic-embed     |
                        +---------------------------+
```

Common stack: **Qwen2.5-Coder-1.5B** for autocomplete, **Qwen3-Coder 32B** (MoE active ~3B) for chat, a small embedding model for retrieval.

## The agent loop

Stripped to essentials:

```
  loop:
    msgs.append({user_message_or_observation})
    completion = call_model(msgs, tools=[...])
    if completion.tool_calls:
        for tc in completion.tool_calls:
            result = execute_tool(tc)
            msgs.append({"role": "tool", "content": result})
    else:
        return completion.content
    if step >= max_steps: break
```

What goes wrong locally:

- **Tool-call schema drift.** Smaller models drop a closing brace or invent a field. Use vLLM-MLX or `outlines` for **schema-constrained generation** so malformed tool calls are impossible by construction.
- **Loops.** A small model will repeat the same tool call. Add a step limiter and a "saw this exact call already" check.
- **Drift on long histories.** Without a summarizer, the chat window blows out. Use a rolling summary every N turns, or delegate summarization to the small fast model while the big model keeps working.

## Tool calls

Three minimum tools for a coding agent:

- `read_file(path)` — bounded by max byte limit.
- `edit_file(path, patch)` — patch format, never full overwrite from a small model.
- `run_shell(cmd, timeout)` — sandboxed; explicit allowlist or workdir restriction.

A more honest list adds `search`, `list_dir`, `web_fetch`, `python_eval`. Each tool's schema lives in JSON, fed to the model's tool-calling layer.

## Streaming for perceived speed

Even when total wall time isn't shorter, streaming tokens to the editor makes the UX feel 2–3× faster. The OpenAI streaming protocol (`data:` SSE chunks) is supported by every engine in Topic 08. Always stream.

## Cloud vs local — the honest comparison

This is **G18 / G19 / G20** of Project 4. Same task in both modes:

| Metric | Cloud (Claude / GPT-4) | Local (this stack) |
|---|---|---|
| TTFT | 350-700 ms | 60-150 ms |
| Total wall (small task) | 4-8 s | 3-6 s |
| Total wall (deep task) | 20-60 s | 30-120 s |
| $ cost | $0.05-0.50 | $0 |
| Privacy | vendor-dependent | on-device |
| Capability ceiling | very high | bounded by your model |
| Variance | low | hardware-dependent |

The right answer is "both, routed." Local for autocomplete and easy tool calls. Cloud or PCC for hard reasoning. The Foundation Models framework already routes this way (Topic 07); your own agent should too.

## Common pitfalls

1. **Picking a 7B for autocomplete.** TTFT budget blown. 1.5B class.
2. **No prefix cache.** Every keystroke reprocesses 200 lines of context. Use the engine's cache and pre-warm it on file open.
3. **Tool-call parsing via regex.** Constrained generation (vLLM-MLX structured output, `outlines`, XGrammar) is the only way to get schema-sane output from small models.
4. **No step limiter.** Local + free + buggy = infinite loop.
5. **Treating tool errors as user-visible.** Wrap and re-feed with hints; do not surface the raw stack trace.

## References

- vLLM structured output: https://docs.vllm.ai/en/latest/features/structured_outputs.html
- outlines: https://github.com/dottxt-ai/outlines
- Continue.dev (open-source local agent reference): https://github.com/continuedev/continue
- Aider local mode: https://github.com/paul-gauthier/aider
- Cursor / Zed / Tabby — the local-first IDEs of 2026
- mlx-lm server: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
- Qwen2.5-Coder family: https://qwenlm.github.io/blog/qwen2.5-coder/
