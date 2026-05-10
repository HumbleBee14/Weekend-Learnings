# 01 — Platform Architecture

## Files

- `CONCEPTS.md` — the five-box mental model, the three reference stacks (vLLM Production Stack, llm-d, NVIDIA Dynamo), control plane vs data plane.
- `architecture.md` — the actual one-page design doc for `mini-platform`. Reuse this verbatim in `mini-platform/architecture.md`.

## Quickstart

There is nothing to run. Read `architecture.md` carefully. Every other topic in this level implements one box from that diagram.

## Try

- Pull up the llm-d architecture page and overlay it on this diagram. Identify which box is the "Endpoint Picker" (EPP) — it's Box 2 (Scheduler). Identify which box is the "Inference Gateway" — it's Box 1.
- Pull up the NVIDIA Dynamo 1.0 blog. Map "Smart Router" -> Box 2, "KVBM" -> Box 4, "SLO Planner" -> Box 5 (autoscaler).
- Now the test: when someone says "Triton Inference Server", which box are they describing? Answer: historically Box 3 (worker), and Triton is now legacy — Dynamo's frontend replaces the management plane around it.

## Where this goes

- Topic 02: Box 5 (control plane) - training job scheduler.
- Topic 03-04: Box 5 - eval + registry.
- Topic 05: Box 5 - observability.
- Topic 06: Box 2 - the router itself.
- Topic 07-09: Box 2 - admission and scheduling policies inside the router.
- Topic 10-11: Box 5 - autoscaling and warmup.
- Topic 12: Box 4 - KV tier.
- Topic 13-15: cross-cutting (cost, safety, reasoning).
- Topic 16: control-plane closure - mini-RLXF.
