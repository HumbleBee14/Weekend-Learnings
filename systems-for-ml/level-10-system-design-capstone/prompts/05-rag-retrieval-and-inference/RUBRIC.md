# Prompt 05 — Rubric

## Strong signals
- Reaches for **Triton Inference Server with ensemble graphs** — the canonical orchestrator for multi-model
- Knows the embedding model is often **CPU-OK** with ORT-INT8 — sized differently from the LLM
- End-to-end latency budget decomposition (embed + retrieve + rerank + LLM)
- Streams the LLM output to mask the synthesis latency

## Anti-signals
- Designs three separate services with HTTP between them — adds 3× round-trip
- Puts the embedding model on a fat GPU — overspend
- No reranker step — degrades retrieval quality, misses Level 5 Topic 13/15 knowledge
- Confuses "RAG" with "fine-tuning" — different system

## What's tested
Multi-model orchestration. This is the only prompt where Triton Inference Server (Level 5 Topic 15) is the headline answer.
