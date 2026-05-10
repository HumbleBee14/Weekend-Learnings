# 14 — Safety and Abuse (at the gateway)

## Why safety belongs at the gateway, not at the model

Three reasons safety logic runs as gateway middleware in 2026, not inside the model serving path:

1. **Defence in depth.** A model alignment failure must not be the only protection. The gateway is *cheap* protection that always runs.
2. **Centralised policy.** One policy update affects every model behind it. No per-engine rebuilds.
3. **Auditability.** Decisions logged at the gateway are machine-readable and queryable; in-model refusals are buried in tokens.

Production runs guardrails as **in-line ext-proc plugins** — same dataplane path as the Inference Scheduler (Topic 06). Envoy AI Gateway, Kong AI Gateway, and the LLM Gateway projects all converge here.

References:
- Envoy AI Gateway guardrails — https://aigateway.envoyproxy.io/docs/capabilities/security/llm-guard/
- NVIDIA NeMo Guardrails — https://docs.nvidia.com/nemo/guardrails/

## The five gateway controls

| Control | Defends against | Where |
|---|---|---|
| Token rate-limits | abuse, cost runaway | gateway (Topic 07) |
| Input filter (prompt classifier / PromptGuard) | prompt injection, jailbreaks | gateway pre-LLM |
| Output filter (Llama Guard / NeMo Guardrails) | unsafe outputs | gateway post-LLM, pre-stream |
| PII / secret redaction | data leak | gateway both directions |
| Abuse counter | repeat offenders | gateway, per-tenant |

### 1. Token rate-limits

Already covered in Topic 07. The point repeated here: token-aware rate-limiting is also the *first* line of abuse defence, because almost every abuse pattern requires high token consumption (jailbreak loops, scraping, exfiltration attempts). A tight per-tenant limit makes pathological behaviour expensive.

### 2. Input filter

Two flavours:

- **Heuristic filters.** Regex / keyword / known-jailbreak-phrase matching. Cheap, fast, low recall. Useful as an *outer ring*.
- **Classifier filters.** Small LLMs (Meta PromptGuard, IBM Granite Guardian, NVIDIA NemoGuardrails) trained to detect prompt injection / unsafe intent. ~10-50ms latency for prompt-level classification. Higher recall.

Production: heuristic ring catches obvious cases without latency cost; classifier ring catches nuanced cases at a small fixed latency.

References:
- Meta PromptGuard — https://ai.meta.com/blog/prompt-guard/
- IBM Granite Guardian — https://huggingface.co/ibm-granite/granite-guardian-3.1-2b

### 3. Output filter

- **Llama Guard 3 / 4.** Meta's content-policy classifier. Reads the assistant's output, returns safe/unsafe + category. ~50-100ms.
- **NVIDIA NeMo Guardrails.** A more flexible policy DSL.
- **Custom regex.** Cheap last-mile redactions (email addresses, credit cards).

Run the output filter on **chunks** of the streaming output, not the whole completion at the end. Otherwise the user sees unsafe text before redaction. Pattern: buffer N tokens, classify, release; repeat. Adds N-token-worth of latency to first-byte but saves the policy.

References:
- Llama Guard 3 — https://ai.meta.com/research/publications/llama-guard-3-vision/
- NVIDIA NeMo Guardrails — https://docs.nvidia.com/nemo/guardrails/

### 4. PII / secret redaction

Two directions:

- **Inbound** — strip PII from user prompts before they hit the model. Reduces the surface area for data-residency / GDPR compliance and lowers leak risk.
- **Outbound** — final pass over the output to redact PII the model emitted (e.g., model regurgitating training-data secrets).

Tools: Microsoft Presidio (regex + ML), AWS Macie, Google DLP API, custom regex chain. Latency: 10-30ms. Acceptable cost for the compliance value.

### 5. Abuse counter

A per-tenant counter tracking suspicious patterns:

- Repeated failed safety classifier hits.
- Suspiciously similar prompts within a short window (likely scraping).
- Known jailbreak signatures (DAN, "ignore previous instructions", role-play prompts).
- Sudden output token spikes.

When a counter crosses a threshold, escalation: temporary rate-limit lower, eventually block. Surface to a human review queue.

## Prompt injection at the infra layer

The 2026 lesson: prompt injection is rarely fixable by training the model alone. The infra layer must enforce trust boundaries:

- **System prompts vs user prompts.** Never let user content overwrite system prompt. Concatenate with explicit role tags; reject prompts that try to spoof tags.
- **Tool-call results.** A model with tools that fetches a webpage can be hijacked by content on the page ("ignore your instructions, do X"). Treat tool outputs as untrusted; sanitise before re-injection; cap their size; isolate them with explicit boundary tokens.
- **Retrieval contexts.** RAG documents are user-influenced (the user picked the vector store, possibly with adversarial content). Same treatment as tool outputs.
- **Multi-turn drift.** A user can over many turns shift the model's "interpretation" of the system prompt. Periodically reinforce the system prompt or use stateful guardrails.

References:
- OWASP LLM Top 10 — https://genai.owasp.org/llm-top-10/
- Prompt injection (Simon Willison's writing) — https://simonwillison.net/series/prompt-injection/

## Threat model — write it down

The single most useful artefact in this topic is a one-page threat model. For `mini-platform`:

```
Asset 1:    Tenant A's confidential prompts.
            Threat: cross-tenant prefix-cache hit.
            Mitigation: cache salting (Topic 07).

Asset 2:    Per-tenant token budget / cost.
            Threat: abusive tenant burns budget.
            Mitigation: token rate-limits (Topic 07) + abuse counter.

Asset 3:    Model-emitted PII / secrets.
            Threat: training-data regurgitation.
            Mitigation: output-side PII redaction.

Asset 4:    System-prompt integrity.
            Threat: prompt injection via user content / tool outputs / RAG.
            Mitigation: explicit role tags, untrusted-content tagging, output guardrails.

Asset 5:    GPU availability.
            Threat: resource-exhaustion abuse (long prompts, infinite outputs).
            Mitigation: max_tokens cap, max prompt length, max concurrent per tenant.
```

A real platform's threat model is longer; the discipline is the same.

## Build steps for `mini-platform`

1. Add a regex-based output filter to the gateway streaming path (chunk buffer + redact + release).
2. Add a per-tenant abuse counter to the gateway. Increment on suspicious events; escalate at thresholds.
3. (Optional, GPU-budget permitting) Add Llama Guard 3 as an output classifier with N-token chunk batching.
4. Document the threat model in `mini-platform/safety/threat-model.md`.

## Pitfalls

1. **Output filter at end-of-stream only.** The user already saw the unsafe output. Filter on chunks.
2. **Single-layer defence.** Heuristic filter alone is pierceable; classifier alone is slow. Stack them.
3. **PII redaction with naive regex.** False negatives are guaranteed; false positives are common. Use Presidio-class tools, augmented with regex.
4. **Ignoring tool outputs as a vector.** Most prompt-injection attacks in agentic systems travel via tool results, not user prompts.
5. **Cache cross-tenant leaks.** Without salting, prefix caching is a covert channel. Topic 07.
6. **No abuse counter at all.** Without it, every offender hits limits forever; you have no escalation signal.

## References

- OWASP LLM Top 10 — https://genai.owasp.org/llm-top-10/
- Llama Guard 3 — https://ai.meta.com/research/publications/llama-guard-3-vision/
- Meta PromptGuard — https://ai.meta.com/blog/prompt-guard/
- IBM Granite Guardian — https://huggingface.co/ibm-granite/granite-guardian-3.1-2b
- NVIDIA NeMo Guardrails — https://docs.nvidia.com/nemo/guardrails/
- Microsoft Presidio — https://microsoft.github.io/presidio/
- Envoy AI Gateway llm-guard — https://aigateway.envoyproxy.io/docs/capabilities/security/llm-guard/
