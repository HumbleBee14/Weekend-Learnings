# Level 10 — ML Systems Design Capstone

> Outer reference: [`systems-for-ml/README.md`](../README.md) · No new project — this level is the *integration capstone* where you whiteboard end-to-end designs using everything from Levels 1–9.
>
> Textbook companion: [Kiely Ch 7 (Production)](../references/Inference-Engineering-Kiely-2025.pdf) + [Reddi Vol 2 *Ops at Scale*](https://mlsysbook.ai/). Both books prep you for the *vocabulary*; this level gives you the *integration practice*.

## Why this level exists

By the end of Level 9 you can build every piece. But building isn't the same as **defending an end-to-end design under questioning** — that's a separate skill, and it's the one that's actually tested in senior ML systems interviews at Anthropic / OpenAI / Together / Anyscale / Scale / Baseten / Databricks / Meta / NVIDIA / etc.

This level exists because:

1. **The hardest interview format for senior ML eng roles is the open-ended system design.** *"Design Bedrock-equivalent for our company"* in 45 minutes. You won't have time to look anything up; the answer comes from internalized intuition built across Levels 1–9.
2. **Real platform work is integration, not invention.** Knowing paged KV cache (L4) and KEDA autoscaling (L7) is different from knowing *which one to spend an hour on when designing a 200-QPS multi-LoRA service.* This level is where you practice that judgment.
3. **There's no public curriculum for this.** Books cover concepts; courses cover techniques; but the *interview-design muscle* gets built by repeated reps against worked-example problems. This level is your gym.

## How to study this level

```
  Day 0 (30m)  ──►  Read this README + the rubric below
  Day 1 (2h)   ──►  Kiely Ch 7 in full — production-eng vocabulary
                  + skim Reddi Vol 2 Ops at Scale
  Day 2 → 6    ──►  Work through prompts 01 → 08 below. For each prompt:
                       1. Read the prompt cold. Set a 45-min timer.
                       2. Whiteboard (paper / iPad / Excalidraw / tldraw) your design.
                       3. Open prompts/0X-name/SOLUTION.md only AFTER your 45 min.
                       4. Compare. Note where you missed components, where you
                          over-engineered, where your $/Mtok math was wrong.
                       5. Re-do the same prompt 3 days later. Time-to-design
                          should drop ~30%.
  Day 7        ──►  Optional: ask a friend to interview you on one of these
                    or record yourself doing one out loud (the hardest test).
```

**Reference order when stuck:**
1. The prompt's `SOLUTION.md` (only after attempting)
2. Kiely Ch 7 (production) — most relevant
3. [CAPACITY-PLANNING.md](../level-7-ml-platform/13-cost-economics/CAPACITY-PLANNING.md) for sizing math
4. [Topic 11b](../level-7-ml-platform/11b-serverless-gpu-substrates/) for substrate decisions
5. Reddi Vol 2 *Ops at Scale* (academic framing)

## The eight canonical prompts

These cover the 80% of senior-ML-eng system-design interview surface area in 2026.

| # | Prompt | What it tests | Levels touched |
|---|---|---|---|
| 01 | Design Bedrock / Vertex AI Inference equivalent | Multi-model serving, multi-tenant fairness, billing | L1, L5, L7 |
| 02 | Design a chatbot at 10K QPS, p99 < 2s | Capacity planning, autoscaling, KV reuse | L4, L5, L7 |
| 03 | Design multi-LoRA serving for 500 fine-tunes / 1 base model | Hot-swap, adapter loading, fairness | L4, L5.10, L7 |
| 04 | Design an inference platform for a 5-engineer startup | Substrate choice (serverless vs K8s), cost/eng tradeoff | L1, L5, L7, L7.11b |
| 05 | Design a RAG-shaped retrieval+inference service | Multi-model pipelines, Triton ensembles, embedding serving | L5.13, L5.15, L7 |
| 06 | Design the inference path for a reasoning model (o1-class) | Long-decode, KV pressure, cancellation propagation | L4.12, L5, L7.15 |
| 07 | Design a distributed training platform for a 70B model | FSDP/TP/PP, NCCL, failure recovery, goodput | L6 (full) |
| 08 | Design a local-first agentic IDE backend (Cursor's local mode) | UMA, MLX, sub-100ms TTFT, privacy | L8 (full) |

Each prompt lives in its own folder (`prompts/01-bedrock-equivalent/`, etc.) with:
- `PROMPT.md` — the question as it would be asked, ~50 words
- `SOLUTION.md` — the worked-out design, ~3-5 pages, with diagrams and the math
- `RUBRIC.md` — what an interviewer is grading on (signals, anti-signals, follow-ups)

## The rubric — what gets evaluated

Senior ML systems design interviews are graded on five axes. Internalize these — they're what separates a 4 from a 3:

```
1. SCOPE & CLARIFY  ── Did you ask the right 3-5 clarifying questions
                      before drawing?  (QPS, latency target, model size,
                      tenancy model, budget?)

2. ARCHITECTURE      ── Can you whiteboard the five boxes (gateway, router,
                       scheduler, worker, control plane) and defend each?

3. QUANTITATIVE      ── Did you do the math?  GPUs needed, $/Mtok, p99 budget
                       decomposition, queue depth, KV memory.  Numbers, not
                       hand-waving.

4. TRADEOFFS         ── Did you name what you're trading off, and what you'd
                       change at 10× scale?  Latency vs throughput vs cost
                       vs complexity vs quality — explicitly.

5. FAILURE & SCALE   ── What happens at 10× the QPS?  What happens when a
                       node dies?  What about a model regression?  Cold
                       starts during a traffic spike?  Did you mention
                       the break-it list?
```

**Anti-signals (instant downgrades):**
- "We'd just use vLLM" — without saying why over SGLang, TRT-LLM
- No numbers anywhere
- Forgot the cost story
- Designed a system that can't fail
- One-replica fault assumption ("if a node dies we just restart it")
- Reaching for Kubernetes for a 5-engineer startup, or serverless for a 200-QPS steady-state production service
- "We'd add caching" without specifying what kind, where, eviction policy, hit-rate target

**Strong signals (3 → 4):**
- Names the specific 2026 production tools (vLLM Production Stack, llm-d, NVIDIA Dynamo, KEDA, OpenTelemetry GenAI semconv)
- Volunteers the failure-injection list before being asked
- Quantifies the cost decision at the design stage
- Knows when *not* to use Kubernetes (Topic 11b)
- Identifies which axis of 5D parallelism this workload needs and why
- References a specific paper or blog when a design choice is non-obvious

## What's *not* in this level

This isn't a coding interview level. No LeetCode. No "implement a paged KV cache" — you already did that in Level 4. The whole level is whiteboard / written design, judged on the rubric above.

This also isn't a "ML modeling" interview level — no fine-tuning strategy, no loss-function debate, no architecture choice (transformer vs SSM). Those are different interviews, different roles.

## Compute

None. Level 10 is paper + whiteboard + maybe Excalidraw. Zero GPU spend.

## Where this fits

- **Comes after:** Levels 1–9 *all of them*. You need the full repo's worth of substrate knowledge to do these prompts well.
- **Comes before:** the interview itself. Or your first day designing a real platform at a new job. Same skill.
- **Project this feeds:** none. This level is the integration capstone; the artifact is your improved design judgment.

## A note on the value of this level

The skill across every JD I sampled (Together AI, Scale, Anthropic, Baseten, GM, Samsara) — is *integration judgment*, not depth in any single component. Levels 1–9 give you depth. Level 10 is where depth becomes judgment.


## References

- Kiely, *Inference Engineering* — [Ch 7 (Production)](../references/Inference-Engineering-Kiely-2025.pdf) — the closest single book to this level
- [Reddi Vol 2 — *Ops at Scale*](https://mlsysbook.ai/)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack) — the open-source reference platform
- [llm-d](https://github.com/llm-d/llm-d) — the CNCF Sandbox reference platform (2026)
- [NVIDIA Dynamo blog posts](https://developer.nvidia.com/blog/) — the production frontier

