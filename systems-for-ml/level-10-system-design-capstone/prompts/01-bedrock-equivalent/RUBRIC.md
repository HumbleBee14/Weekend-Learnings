# Prompt 01 — Rubric

## Strong signals (3 → 4)
- Names KV-cache-aware routing (not just "round-robin")
- Reaches for **NVIDIA Dynamo or llm-d** for the 70B-class — the 2026 frontier
- Specifies OpenTelemetry GenAI semconv for the metering pipeline (not "we'll log stuff")
- Handles cold-start during region failover (cross-region warm pools)
- Has a per-tenant WFQ + token-aware rate-limit story
- Mentions disaggregated prefill/decode for 70B-class — and *why* (decode-vs-prefill GPU specialization)
- Quantifies SLA — 99.95% means ~22 minutes downtime per month per region; can your design tolerate that?

## Anti-signals (instant 2)
- "Just put it behind an ALB" — at this scale, you need an LLM-aware gateway
- No billing pipeline mentioned — this is half the system for a managed offering
- One global region — the prompt says 5 regions
- All-vLLM, no Dynamo for 70B — leaves performance on the table at this scale
- No multi-tenancy story — "we'd just hash-partition" isn't an answer for paying customers

## What's really being tested
Integration depth across the full Level 7 stack. This prompt deliberately leaves no room for clever substrate shortcuts — you have to actually know the production stack.
