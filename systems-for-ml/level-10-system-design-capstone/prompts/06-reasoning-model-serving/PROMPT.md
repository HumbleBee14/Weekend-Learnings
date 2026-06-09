# Prompt 06 — Reasoning Model Serving (o1-class)

Your company is launching a reasoning-mode product. The model is a 70B-class reasoning model (think DeepSeek-R1, o1, Claude with extended thinking). Distinctive characteristics:

- **Output length is highly variable:** mean 2000 tokens, p99 30,000 tokens
- **Users wait long** — p95 end-to-end can be 60+ seconds; that's accepted UX
- Output is **mostly the reasoning trace**; the final answer is short
- Cancellation matters — users might abandon after 20 seconds if they see no progress
- 20 QPS sustained, projected 100 QPS in 6 months

Design the inference platform. 45-minute interview format.
