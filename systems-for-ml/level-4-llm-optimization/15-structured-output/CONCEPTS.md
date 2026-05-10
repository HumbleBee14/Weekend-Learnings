# 15 — Structured Output

## What it is

Force the model's output to conform to a grammar — JSON schema, regex, BNF. Implemented by *masking the logits* at each step to allow only valid next tokens.

```
Without grammar:  next-token probabilities cover all 150,000 vocab tokens
With grammar:     next-token probabilities are zeroed for tokens that don't match the
                  grammar's allowed set at this position; sampling is constrained
```

The model can't produce "Once upon a time" if you've asked for valid JSON. Wrong tokens have probability 0; the model picks among valid tokens only.

This is critical for agentic workloads in 2026. If your agent expects a `{"action": "click", "target": "..."}` response, "I think we should click..." is unrecoverable. Grammar masking guarantees parseable output.

## The 2026 standard: xgrammar

In 2024, **Outlines** was dominant. By 2026, **xgrammar** has consolidated as the default backend for vLLM, SGLang, TensorRT-LLM, and MLC-LLM.

What changed:

- Compilation of the grammar to a finite-state machine is much faster
- Cross-grammar caching (reuse compiled FSMs across requests with the same schema)
- Near-zero per-token overhead in serving (was 5-15% with Outlines)

**XGrammar-2 (May 2026)** — released a few weeks ago — pushes this further:

- Up to 80× compilation speedup over XGrammar (cross-grammar caching)
- Repetition-state compression
- Batch + spec-decode support (was a gap in XGrammar 1)

## How grammar masking works

```
Step 1: Generate logits for next token (full vocab)
Step 2: Compute mask from grammar state:
          mask[token] = 1 if token can validly follow current state, else 0
Step 3: logits = logits + log(mask)   # invalid tokens get -inf
Step 4: Sample as usual (top-k, top-p, temperature)
Step 5: Update grammar state with the chosen token
Step 6: Repeat
```

The grammar state is a position in a finite-state machine derived from the schema. Each transition allows certain tokens; the mask reflects allowed transitions.

## Performance characteristics

- **Reused schemas**: near-zero overhead. The FSM compiles once, mask computation is fast.
- **Unique schemas per request**: TTFT suffers because each request triggers fresh FSM compilation.
- **Long valid token sets**: cheap (mostly the model picks naturally-valid tokens; mask doesn't bite).
- **Highly constrained schemas** (very narrow valid sets): can hurt quality if the constraint is unnatural for the model.

## What you can constrain

- **JSON schema** — full JSON Schema spec; the everyday case
- **Regex** — for simple patterns
- **BNF / Lark grammar** — arbitrary context-free grammars
- **Choice constraints** — pick one of N strings (function-calling style)

## OpenAI-compatible structured outputs

vLLM, SGLang, TGI all support the OpenAI-compatible API:

```python
client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["click", "type", "navigate"]},
                    "target": {"type": "string"},
                },
                "required": ["action", "target"],
            }
        }
    }
)
```

vLLM picks the right backend automatically (`auto` mode in 2026).

## Outlines and LMFE — the alternatives

- **Outlines** — pioneered FSM-based structured generation. Still alive, less common in vLLM-centric stacks.
- **LMFE (LM Format Enforcer)** — niche, mostly seen in research code.

For new code: use xgrammar via vLLM/SGLang's API. Don't reach for Outlines unless you have a specific reason.

## Pitfalls

1. **Forgetting to enable structured output for tool calls.** Many production failures: model "kind of" produces JSON, parser fails. Always use grammar masking for any parseable output.
2. **Highly constrained schema regressing on quality.** If the grammar is too restrictive (e.g., enum with values the model doesn't naturally produce), the model gets confused. Verify with eval.
3. **Treating XGrammar overhead as zero.** It's near-zero with cached schemas; can be a few percent of TTFT with novel schemas every request. If your schema is request-specific, cache aggressively.
4. **Grammar masking + spec decode interaction**. XGrammar 1 didn't support spec decode well; XGrammar-2 does. If your stack combines both, use XGrammar-2.
5. **Skipping schema validation on the model side.** Always validate the JSON output on receive, even with grammar masking. Belt-and-suspenders.

## What you'll do

Add structured output to your `mini-vllm` (or use vLLM directly):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "Find a restaurant in Berlin."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                    "rating": {"type": "number", "minimum": 0, "maximum": 5},
                },
                "required": ["name", "city", "rating"]
            }
        }
    }
)
```

Measure:

- ITL with vs without grammar masking. Should be near-identical with cached schemas.
- TTFT with novel schemas per request. Slightly higher.
- Failure rate of JSON parsing on receive. Should be 0 with grammar masking; 10-30% without on hard prompts.

## References

- xgrammar — https://xgrammar.mlc.ai/
- XGrammar-2 release (May 2026) — https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation
- vLLM structured outputs docs — https://docs.vllm.ai/en/latest/usage/structured_outputs.html
- Outlines — https://github.com/outlines-dev/outlines
- LMFE — https://github.com/noamgat/lm-format-enforcer
