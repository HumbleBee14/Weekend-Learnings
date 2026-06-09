# Prompt 06 — Rubric

## Strong signals
- Identifies **separate pool for reasoning** vs chat — mixed continuous batching starves short requests
- **Cancellation propagation** as a first-class concern with concrete latency target
- **KV-budget-per-request** as the binding constraint (not throughput)
- **Streaming the reasoning trace** to mask the wait
- Spec decode being *more* valuable for long-decode workloads

## Anti-signals
- "Treat reasoning like any other LLM request" — misses every constraint
- No cancellation story — wastes 40s of GPU on every user abort
- No output budget — one user can rack up $10 of GPU on a single request
- Doesn't size by KV memory

## What's tested
Knowledge that reasoning workloads break the normal serving design. Topic 15 of Level 7 exists exactly for this.
