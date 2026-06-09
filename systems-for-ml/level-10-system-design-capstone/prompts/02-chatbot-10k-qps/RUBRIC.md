# Prompt 02 — Rubric

## Strong signals
- Recognizes this is a **prefix-cache workload** (multi-turn chat) → SGLang RadixAttention
- Reaches for **FP8 + spec decode + disagg + prefix cache** as a compounding stack, with $$ math
- Knows decode dominates at this output-shape and sizes for it
- Has a region-failover story for 100M MAU scale

## Anti-signals
- Defaults to vLLM without justifying over SGLang for multi-turn chat
- No quantization story — leaves 30% on the table
- Single-region design
- No spec-decode mention

## What's being tested
Optimization-stack composition at scale. The compound effect of L4 techniques. The judgment to know which optimization matters for *this* workload shape.
