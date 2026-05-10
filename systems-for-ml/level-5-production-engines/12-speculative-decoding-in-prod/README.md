# 12 — Speculative Decoding in Production

## Files

- `CONCEPTS.md` — the 2026 spec landscape (n-gram / EAGLE-3 / P-EAGLE / MTP), acceptance rate as the metric, systems-level interactions, when spec decode hurts
- `measure_spec.py` — drives a baseline vLLM server and a spec-on vLLM server with the same chat / code / reasoning workload and reports speedup + acceptance rate

## Quickstart

Two servers on different ports:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8011 \
    --speculative-config '{"method":"ngram","prompt_lookup_max":4,"num_speculative_tokens":4}'

pip install openai httpx
python measure_spec.py --baseline http://localhost:8010 --spec http://localhost:8011
```

## Expected output

```
category    baseline tok/s   spec tok/s   speedup   TTFT base ms   TTFT spec ms
chat                   210          340      1.62x            220            245
code                   195          355      1.82x            215            240
reasoning              200          205      1.03x            220            250

Server-reported acceptance rate (cumulative): 71.3%
```

The reasoning row should be near 1× — n-gram acceptance is low on hard-reasoning text.

## Try

- **Try EAGLE-3** if a head exists for your model. Acceptance should jump to 75-85% on chat/code.
- **Quality check.** Run `lm-eval-harness` against both servers. Scores must match within noise; if they don't, the spec implementation has a sampling bug.
- **High concurrency.** Re-run with concurrency 16. Spec wins less per-request but should still help aggregate throughput unless you're saturating verify-pass capacity.
- **Disable spec on reasoning, enable on chat.** Per-route control is what real production stacks do — Level 7's router should know.

## Where this goes

- Project 2 — spec-decode lift is one of the per-engine numbers in the bake-off
- Level 4 Topic 13 — algorithm-level treatment if you want to revisit
- Level 7 — per-tenant / per-route spec-decode policy as a router decision
