# 15 — RL Post-Training Bridge

## Files

- `CONCEPTS.md` — the two-engine pattern (trainer + rollout engine), 2026 framework landscape, where this topic plugs into Level 7.
- `HANDS-ON.md` — runnable ~30-min smoke test: TRL GRPO + vLLM as rollout backend on a 0.5B model. Lets you *feel* the trainer↔rollout handoff in working code.

The full RLHF curriculum is its own thing; the goal here is (1) reading-level fluency on the architecture, (2) one runnable example that exercises Level 5's vLLM engine in its post-training role, and (3) recognition of where Level 6's primitives (NIXL, async DCP, FSDP2) reappear.

## Quickstart

Read `CONCEPTS.md` first. Then either:

**Path A — runnable smoke test (recommended).** Follow [`HANDS-ON.md`](HANDS-ON.md). Requires 1 GPU with ≥12GB. ~30 min. You'll watch rollout time dominate training time — the empirical fact that motivates every production RL framework.

**Path B — read-only.** Skim:
- verl HybridFlow paper: [arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)
- DeepSeek-R1 (GRPO at scale): [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- TRL GRPO trainer: [huggingface.co/docs/trl/main/en/grpo_trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)

## Boundary check

The hands-on file is a **smoke test, not a research run.** The reward function is silly; 20 steps don't produce a useful model. Real RLHF (reward modeling, KL constraints, off-policy correction, multi-node rollout pools) belongs to a separate curriculum. Stay in the bridge-topic budget.

## Where this goes

- Level 7's `mini-platform` — `mini-rlxf` demo will glue trainer + vLLM rollout
- Level 7's serving-side reuses the same inference engine as a rollout backend; the only thing that changes is who's calling it
