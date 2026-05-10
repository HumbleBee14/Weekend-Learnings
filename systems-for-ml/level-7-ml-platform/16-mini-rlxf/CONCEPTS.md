# 16 — mini-RLXF

## RLXF as a platform problem, not a research problem

Reinforcement learning from feedback (RLHF, RLAIF, PPO, GRPO, DPO, ...) is the most platform-shaped piece of training. Why:

```
trainer  <───  weights ────  rollout (vLLM/SGLang)
  │                                 │
  │                              prompts
  │                                 │
  │                                 ▼
  │                          reward model (RM)  /  rule-based reward
  │                                 │
  │                              rewards
  │                                 │
  ▼                                 ▼
optimizer step <─── replay buffer ──┘
```

Four concurrent components. The trainer holds the weights; the rollout serves them at high throughput; the reward model scores; the buffer brokers between them. Every interesting failure is at a *boundary* between two components.

In 2026 the production stack is opinionated:

| Role | Production tool |
|---|---|
| Trainer | verl (most adopted), OpenRLHF, NeMo-RL, TRL |
| Rollout | vLLM or SGLang (rare to roll your own) |
| Reward | reward LLM via vLLM, or rule-based |
| Buffer | in-memory or Redis Streams |
| Weight sync | NCCL broadcast directly into the rollout's HBM |

References:
- verl — https://github.com/volcengine/verl
- OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
- NeMo-RL — https://github.com/NVIDIA/NeMo-RL
- TRL — https://huggingface.co/docs/trl
- DeepSeek GRPO paper — https://arxiv.org/abs/2402.03300

## Why rollouts run on vLLM, not on the trainer

A naive RL loop runs `model.generate(...)` inside the training framework. Two reasons that's a non-starter at any scale:

1. **Throughput.** `transformers.generate` is single-stream, no continuous batching, no paged KV. A vLLM rollout is 10-50x faster for the same workload. With long episodes, the trainer would spend 90% of wall-clock generating and 10% optimising.
2. **Resource isolation.** Rollouts and gradient computation want different memory profiles. Rollouts want max KV; training wants max activation memory. Sharing one process forces compromises.

The 2026 default: rollout runs on vLLM workers with **NCCL weight sync** every N optimisation steps. The trainer pushes new weights into the rollout's HBM via NCCL `broadcast` from a designated rank — no roundtrip to disk.

## NCCL weight sync — the load-bearing piece

```
trainer rank 0 (DP master)            rollout vLLM worker
  │                                       │
  ├── after step N:                       │
  │   gather full weights                 │
  │   ncclBroadcast(weights) ────────────►│ ncclBroadcast(weights, root=trainer)
  │                                       │ swap into engine.model
  │                                       │ (vLLM exposes update_weights() in V1+)
  ▼                                       ▼
  continue training                       continue rollouts with new policy
```

The transfer is GPU-to-GPU over NVLink/RDMA — same machinery as Level 6's data parallelism, repurposed. For a 7B model in BF16 (~14 GB), the broadcast is sub-second on InfiniBand, low single-digit seconds on Ethernet.

vLLM exposes `engine.update_weights()` for hot-swap. SGLang has the same primitive.

## On-policy vs off-policy in this loop

- **On-policy (PPO/GRPO).** The rollout uses the *current* policy weights. After each optimiser step, weight sync is mandatory. Latency-sensitive: weight-sync slow path = trainer idle.
- **Off-policy (DPO/RLAIF on a fixed corpus).** Rollouts are generated once, optimisation runs on the static buffer. No weight sync at all.

GRPO has become the most-adopted RL algorithm in 2026 (DeepSeek originated it, Kimi K2 / Qwen3 used variants). It's an on-policy method; weight sync is the critical-path operation. Get it right.

## Reward sources

Three flavours, often combined:

1. **Rule-based.** Check correctness on math/code tasks. Cheap, deterministic. Used in DeepSeek R1's main training.
2. **Reward model (RM).** A trained classifier or scoring LLM. More flexible, more expensive, requires its own training pipeline.
3. **Rubric-based / LLM-as-judge.** A strong frozen LLM scores rollouts on a rubric. Most flexible; quality bound by judge.

Rule-based is the cleanest start. Hard tasks (math problems with verifiable answers, code with unit tests) are the natural fit; the loop becomes generate-and-grade.

## The orchestration story

This is where the rest of Level 7 is pulled in:

| Topic | Role in mini-RLXF |
|---|---|
| 02 (scheduler) | submits training jobs and rollout-batch jobs |
| 03 (eval) | gates "approved" RL checkpoints before serving |
| 04 (registry) | versions every RL checkpoint, tracks reward curves |
| 05 (observability) | per-step reward / KL / policy-entropy panels |
| 06 (router) | rollout requests are LLM requests; KV-aware routing applies |
| 07 (fairness) | rollout traffic gets its own tenant + WFQ weight |
| 10 (autoscaling) | rollout pool autoscales separately on its own queue depth |
| 16 (this) | the loop wiring |

That is *the* point of doing mini-RLXF inside the platform week — it's the most demanding integration of every other piece.

## Build steps (light touch)

Goal: show the architecture in working form. Not to converge a model.

1. Trainer = your Level 6 setup (FSDP2 on a small base, e.g. Qwen3-1.7B).
2. Rollout = vLLM serving the same base, registered in your registry as `model=base, status=serving`.
3. Reward = rule-based on a small math benchmark (GSM8K).
4. Wire weight sync: after every M trainer steps, gather full weights at rank 0, broadcast to vLLM engine, call `update_weights()`.
5. Run a few hundred RL steps; capture reward curve and KL divergence vs base.
6. Document the architecture in `mini-platform/rlxf/`. Convergence is not the deliverable.

## Pitfalls

1. **Single process for trainer + rollout.** Throughput dies. Always split.
2. **Disk-roundtrip weight sync.** 30s+ per step. Must be NCCL.
3. **Stale rollouts on on-policy methods.** GRPO/PPO break if rollouts lag too far behind weights. Sync cadence matters.
4. **Reward hacking.** A reward model becomes a target the policy gradients toward, not the underlying intent. Rule-based rewards are robust here.
5. **No KL bounding.** Policy collapses or diverges from base. Always include a KL-to-reference penalty (PPO clip / GRPO KL term).
6. **Ignoring rollout cost.** RLXF easily spends 80% of wall-clock on rollouts. Treat the rollout pool as the binding constraint, not the trainer.

## References

- verl — https://github.com/volcengine/verl
- OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
- NeMo-RL — https://github.com/NVIDIA/NeMo-RL
- TRL — https://huggingface.co/docs/trl
- GRPO (DeepSeek-R1) — https://arxiv.org/abs/2501.12948
- vLLM weight update API — https://docs.vllm.ai/en/latest/
