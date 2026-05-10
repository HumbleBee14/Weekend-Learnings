# 14 — Safety and Abuse

## Files

- `CONCEPTS.md` — why safety belongs at the gateway, the five gateway controls, prompt injection at the infra layer, threat-model discipline.
- `safety_middleware.py` — `InputFilter`, chunked `OutputFilter`, `AbuseCounter`. Drop-in for Topic 06's router.
- `threat-model.md` — one-page threat model for `mini-platform`, ready to copy under `mini-platform/safety/`.

## Quickstart

```bash
python safety_middleware.py
```

## Expected output

```
(True, None)
(False, 'jailbreak pattern: ignore (all )?previous (instructions|prompts)')
Filtered output: Here is your secret. API key: [REDACTED], email: [REDACTED]
hit 1: ok
hit 2: ok
hit 3: degrade
hit 4: degrade
hit 5: degrade
hit 6: block
hit 7: block
hit 8: block
```

## Try

- **Wire to Topic 06.** Call `InputFilter.check` before `router.pick`; wrap the upstream `aiter_raw` stream with `OutputFilter.feed/flush`; increment `AbuseCounter` on safety hits.
- **Llama Guard.** Replace the regex output filter with a real Llama Guard 3 call on chunk boundaries. Measure added TTFT cost.
- **Tool-call boundary.** Add a `<|untrusted|>...<|/untrusted|>` wrapping convention for tool outputs in your agent loop. Confirm a "ignore previous instructions" payload inside the wrapper does not change behaviour.
- **PromptGuard.** Add Meta PromptGuard as the second-ring input filter and measure the recall delta vs regex-only on a small jailbreak-prompt corpus.

## Where this goes

- Topic 07: token rate limits + abuse counter together are the cost-runaway defence.
- Topic 15: cancellation propagation is itself a safety primitive — abandoned long generations stop costing GPU-seconds and stop emitting tokens that might be harmful.
- `reports/platform.md`: include `threat-model.md` as an appendix or section.

## References

- OWASP LLM Top 10 — https://genai.owasp.org/llm-top-10/
- Llama Guard 3 — https://ai.meta.com/research/publications/llama-guard-3-vision/
- Meta PromptGuard — https://ai.meta.com/blog/prompt-guard/
- NVIDIA NeMo Guardrails — https://docs.nvidia.com/nemo/guardrails/
- Microsoft Presidio — https://microsoft.github.io/presidio/
