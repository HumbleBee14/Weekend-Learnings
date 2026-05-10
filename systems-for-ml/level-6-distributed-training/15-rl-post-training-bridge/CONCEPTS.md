# 15 — RL Post-Training Bridge

Brief topic. The 2026 reality: RLHF/GRPO/PPO training requires generating completions during training (rollouts). Those rollouts run on **vLLM or SGLang** — not on the training framework. The training framework holds the trainable copy of the model; the inference engine holds a fast-serving copy and produces samples. This is the architecture that bridges Level 6's training story to Level 5's serving story.

## Why rollouts run on inference engines

- Training framework forwards are slow at autoregressive generation: no paged KV, no continuous batching, no FlashInfer.
- vLLM/SGLang run the same model 5–20× faster for generation.
- During RL training you need *thousands* of completions per step. Rollout-engine speed is the gating factor.

## The two-engine pattern

```
   ┌────── trainer (FSDP / Megatron / torchtitan) ──────┐
   │  forward+backward on training samples              │
   │  optimizer step                                    │
   │  every K steps: weight sync → rollout engine       │
   └─────────────────┬──────────────────────────────────┘
                     │ weight transfer (NCCL or NIXL)
                     ▼
   ┌────── rollout engine (vLLM / SGLang) ──────────────┐
   │  receives prompts                                  │
   │  generates completions at high throughput          │
   │  ships completions back to trainer                 │
   └────────────────────────────────────────────────────┘
```

The weight-sync step is the systems-interesting part. Two patterns:
- **NCCL broadcast**: the trainer's FSDP-sharded weights are gathered and broadcast to the rollout engine's TP-sharded layout. Custom resharding logic.
- **NIXL** (Topic 00b): point-to-point HBM-to-HBM transfer of weight tensors. More flexible than collectives, lower overhead for the resharding case.

## Frameworks

In 2026 the active stacks:

- **verl** ([github.com/volcengine/verl](https://github.com/volcengine/verl), HybridFlow architecture) — increasingly used in production RLHF. Composes vLLM/SGLang as rollout backend with FSDP/Megatron as trainer.
- **OpenRLHF** ([github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)) — popular open-source. Ray-backed.
- **TRL** ([github.com/huggingface/trl](https://github.com/huggingface/trl)) — HuggingFace, smaller-scale. SFT, DPO, GRPO.
- **NeMo-RL** — NVIDIA stack.

## Algorithms (2026 dominant)

- **GRPO** (DeepSeek): no value model; uses group-relative advantages. Cheaper to train than PPO. Most active research direction.
- **DPO / IPO**: offline preference optimization. Doesn't need rollouts during training. Used heavily for alignment tuning.
- **PPO**: still in production at frontier labs but seen as legacy by many.

## Co-located vs disaggregated rollouts

- **Co-located**: trainer and rollout engine on the same nodes, time-shared. Rollout phase frees compute → training phase uses it. Simple, but throughput-suboptimal.
- **Disaggregated**: separate worker pools. Trainer streams prompts to rollout pool; rollout pool streams completions back. Higher steady-state utilization, more orchestration complexity. The preferred pattern at frontier scale.

## Why this is just a bridge topic

You don't run RLHF this week — that's its own multi-week curriculum. The point here:
- Know the architecture (training engine + rollout engine + reward model + reference model).
- Know which inference engine you'd reach for as rollout backend (vLLM is most common; SGLang for structured-output rollouts).
- Recognize that NIXL (Topic 00b) and async DCP (Topic 13) both show up here — same primitives, different use case.

Level 7's `mini-platform` will host a simple GRPO demo if time permits (the `mini-rlxf` topic).

## Reference

- verl: [github.com/volcengine/verl](https://github.com/volcengine/verl)
- HybridFlow paper: [arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)
- OpenRLHF: [github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- TRL: [github.com/huggingface/trl](https://github.com/huggingface/trl)
- DeepSeek-R1 (GRPO at scale): [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- NeMo-RL: [github.com/NVIDIA/NeMo-RL](https://github.com/NVIDIA/NeMo-RL)
