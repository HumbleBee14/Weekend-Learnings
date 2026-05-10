# 09 — Dynamo and llm-d

## Files

- `CONCEPTS.md` — what Dynamo is, what llm-d is, how they compare, why it matters for the curriculum
- `architecture_compare.py` — prints an ASCII side-by-side and the mapping to Level 7 topics

## Quickstart

```bash
python architecture_compare.py
```

Then read in this order (90 min total):

1. NVIDIA Dynamo overview — https://docs.nvidia.com/dynamo/latest/
2. llm-d overview — https://llm-d.ai/docs/
3. vLLM Production Stack quickstart (the simplest end-to-end you can actually run) — https://docs.vllm.ai/projects/production-stack/en/latest/

## Try

- **Run the vLLM Production Stack on a Kind cluster** (Kubernetes in Docker). It boots vLLM + LMCache + Envoy AI Gateway locally on a Mac. You see the real shapes (pods, services, CRDs) without renting a fleet.
- **Inspect Envoy AI Gateway's extProc plugin** — that's where KV-cache-aware routing logic lives in llm-d.
- **Read NIXL's source** — https://github.com/ai-dynamo/nixl. ~5K lines of C++; the KV transport primitive everyone is building on.

## What to walk away with

- A one-page diagram of either Dynamo or llm-d, with each component labeled.
- A list of which components map to which Level 7 mini-platform topics.
- The vocabulary: NIXL, LMCache, InferencePool CRDs, Envoy AI Gateway, NIM. When these come up in 2026 papers and blog posts, you know what's being talked about.

## Where this goes

- Level 7 — your `mini-platform` is the toy version of these. By Level 7's end, every component above has a working toy implementation in your repo.
