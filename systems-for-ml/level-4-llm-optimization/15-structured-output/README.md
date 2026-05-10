# 15 — Structured Output

## Files

- `CONCEPTS.md` — what grammar masking does, xgrammar (the 2026 default), XGrammar-2 (May 2026), how grammar interacts with sampling, performance characteristics
- `test_grammar.py` — call vLLM's OpenAI-compatible API with and without JSON schema; measure parse-validity and time

## Quickstart

```bash
pip install vllm openai
vllm serve Qwen/Qwen2.5-1.5B-Instruct &
sleep 30   # wait for vLLM to load
python test_grammar.py
```

## Expected output

```
=== Without grammar masking ===
  [1/5] valid_json=False  time=2.34s
      I'd recommend Restaurant Tim Raue in Berlin...
  [2/5] valid_json=True   time=2.41s
      {"name": "Tim Raue", ...}
  [3/5] valid_json=False  time=2.38s
      Sure! Here's a great recommendation: ...
  [4/5] valid_json=True   time=2.45s
  [5/5] valid_json=False  time=2.31s

  → 2/5 produced valid JSON
  → median time: 2.38s

=== With JSON schema grammar masking ===
  [1/5] valid_json=True  time=2.42s
      {"name":"Restaurant Tim Raue","city":"Berlin","rating":4.7,...}
  [2/5] valid_json=True  time=2.39s
  [3/5] valid_json=True  time=2.44s
  [4/5] valid_json=True  time=2.41s
  [5/5] valid_json=True  time=2.37s

  → 5/5 produced valid JSON (should be 100%)
  → 5/5 schema-compliant
  → median time: 2.41s
```

The headline: 100% parse-validity vs ~40% without. Time difference is within noise.

## Try

- **Add `enum` constraints** to fields (e.g., `"action": {"type": "string", "enum": ["click", "type"]}`). The model is forced into one of those values.
- **Make the schema very narrow** — see if quality regresses (model can't say what it wants to say).
- **Send 100 requests with different schemas** vs 100 with the same schema. Measure TTFT delta. Cache effects matter.
- **Combine with spec decode** — make sure XGrammar-2 backend (not XGrammar 1) is used when both are enabled.

## What you should walk away with

- Working JSON schema constraint via vLLM API
- Demonstrated parse-validity win (the production-relevant metric)
- Awareness of XGrammar-2 (May 2026) for combined grammar + spec decode
- Understanding that grammar masking is "almost free" with cached schemas

## Where this goes

- Topic 16 — serving concurrency primitives
- Topic 17 — speculative decoding systems integration
