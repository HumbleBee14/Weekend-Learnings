# 15 — RL Post-Training Bridge

## Files

- `CONCEPTS.md` — the two-engine pattern (trainer + rollout engine), 2026 framework landscape, where this topic plugs into Level 7

No code in this folder by design. RLHF is its own curriculum; the goal here is reading-level fluency on the architecture and where Level 6's primitives (NIXL, async DCP, FSDP2) reappear.

## Quickstart

Read `CONCEPTS.md`. Then skim:

- verl HybridFlow paper: [arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)
- DeepSeek-R1 (GRPO at scale): [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- TRL GRPO trainer: [huggingface.co/docs/trl/main/en/grpo_trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)

## Try (optional, light)

If you have a small base model from Topic 10 and want to taste RLHF:

```bash
pip install trl
# write a small reward function (e.g., length, simple regex)
# run TRL's GRPO on the checkpoint
```

But the real RLHF practice belongs to a separate weekend. Don't burn time here — this topic is the conceptual bridge to Level 7.

## Where this goes

- Level 7's `mini-platform` — `mini-rlxf` demo will glue trainer + vLLM rollout
- Level 7's serving-side reuses the same inference engine as a rollout backend; the only thing that changes is who's calling it
