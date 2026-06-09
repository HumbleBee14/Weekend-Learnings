# Eight canonical prompts

Each prompt has its own folder with:
- `PROMPT.md` — the question as it would be asked, ~50 words. Read this cold, set a timer.
- `SOLUTION.md` — the worked design, ~3-5 pages. Open *only after* attempting.
- `RUBRIC.md` — what an interviewer grades on for this specific prompt.

## The set

| # | Prompt | Hardest part | Levels touched |
|---|---|---|---|
| [01](01-bedrock-equivalent/) | Bedrock / Vertex-AI-Inference equivalent | Multi-tenant billing + isolation | L1 + L5 + L7 |
| [02](02-chatbot-10k-qps/) | Chatbot, 10K QPS, p99 < 2s | Capacity planning + KV reuse | L4 + L5 + L7 |
| [03](03-multi-lora-500-tunes/) | 500 LoRAs, 1 base model, hot-swap | Adapter routing + fairness | L4 + L5.10 + L7 |
| [04](04-startup-5-engineers/) | Inference platform for a 5-eng startup | Substrate choice (PaaS vs K8s) | L1 + L5 + L7 + L7.11b |
| [05](05-rag-retrieval-and-inference/) | RAG service, multi-model pipeline | Triton ensembles + embedding serving | L5.13 + L5.15 + L7 |
| [06](06-reasoning-model-serving/) | Reasoning model (o1-class) serving | Long decode + KV pressure | L4.12 + L5 + L7.15 |
| [07](07-distributed-training-70b/) | Distributed training, 70B model | Parallelism dimensionality (2D→3D) + goodput | L6 (full) |
| [08](08-local-agentic-ide-backend/) | Local-first agentic IDE backend | UMA + sub-100ms TTFT | L8 (full) |

## How to use this folder

1. **Pick a prompt.** Start with 04 (startup, 5 engineers) — it's the most realistic 2026 scenario and pulls in substrate choice (Topic 11b) which is the freshest piece.
2. **Set a 45-min timer.** Don't peek at the solution.
3. **Whiteboard your design.** Paper, iPad, Excalidraw, tldraw — whatever works. Aim for the five boxes (gateway, router, scheduler, worker, control plane) labeled clearly.
4. **Open `SOLUTION.md`** when the timer goes off. Compare.
5. **Note the gaps.** Where did you miss components? Where was your math sloppy? Where did you over-engineer?
6. **Re-do the same prompt 3 days later.** Watch your time-to-design drop ~30%. Watch your number of clarifying questions go up.

## Why eight, not twelve

Eight is the smallest set that covers the design surface area for senior ML systems roles in 2026: serving (01–05), reasoning (06), training (07), and on-device (08). Adding more prompts adds variance, not coverage. Repetition on these eight is what builds the muscle, not collecting twenty.

## What's in each SOLUTION.md

All eight solutions are at full worked depth, following the same 8-section template:

1. **Clarifying questions** (3-5 of them, the kind a senior candidate asks before drawing)
2. **The right answer in one sentence** (opinionated, defensible)
3. **The architecture** (ASCII whiteboard + the five-box mapping)
4. **Capacity math** (concrete GPU counts, $/Mtok, p99 headroom decomposition)
5. **The hard parts** (3-5 differentiating beats for *this* prompt specifically)
6. **Break-it list** (failure modes + mitigations as a table)
7. **What changes at 10× scale** (the seniority signal)
8. **The 30-second summary** + "What this prompt is really testing" + cross-references

Solutions name specific 2026 production tools (vLLM Production Stack, NVIDIA Dynamo 1.0, llm-d, LMCache, Envoy AI Gateway, KEDA, EAGLE-3, MI300X, B200, Modal, MLX). Use the same vocabulary in your interview answer.
