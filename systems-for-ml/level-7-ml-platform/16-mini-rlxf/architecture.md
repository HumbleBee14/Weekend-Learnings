# `mini-platform/rlxf` — architecture sketch

## The loop

```
                    ┌─────────────────────────────────────┐
                    │ 1. Sample batch of prompts          │
                    │    (math problems, with answers)    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ 2. ROLLOUT (vLLM workers)           │
                    │    - KV-aware routing (Topic 06)    │
                    │    - generate K samples per prompt  │
                    │    - return token log-probs         │
                    └──────────────┬──────────────────────┘
                                   │ rollouts
                    ┌──────────────▼──────────────────────┐
                    │ 3. REWARD                            │
                    │    rule-based: parse answer, check   │
                    │    against ground-truth.             │
                    └──────────────┬──────────────────────┘
                                   │ rewards
                    ┌──────────────▼──────────────────────┐
                    │ 4. ADVANTAGE / GROUP NORMALISATION   │
                    │    GRPO: (r - mean(r_group)) / std   │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │ 5. TRAINER step (FSDP2, Level 6)     │
                    │    PPO/GRPO loss, KL-to-reference    │
                    │    clip                              │
                    └──────────────┬──────────────────────┘
                                   │ every M steps
                    ┌──────────────▼──────────────────────┐
                    │ 6. WEIGHT SYNC                      │
                    │    gather full weights at rank 0     │
                    │    NCCL.broadcast -> vLLM HBM        │
                    │    engine.update_weights()           │
                    └──────────────┬──────────────────────┘
                                   │
                                   └─► back to step 1
```

## Where it touches the rest of Level 7

- Topic 02 schedules trainer jobs and rollout-batch jobs.
- Topic 04 registers each RL checkpoint with reward curve metadata.
- Topic 05 dashboards: reward, KL, policy entropy, rollout TPS.
- Topic 06 router serves rollouts (the rollout pool is "just" another LLM client).
- Topic 07 puts rollouts in their own tenant lane with high WFQ weight (bursty, must not crowd interactive traffic).
- Topic 10 autoscales the rollout pool independently from the interactive pool.

## What you ship in `mini-platform/rlxf/`

- `trainer.py` — FSDP2 trainer (carried from Level 6) extended with the PPO/GRPO loss.
- `rollout_client.py` — async client that hits vLLM, returns rollouts + log-probs.
- `reward_rule.py` — math-grading rule reward (GSM8K-shaped).
- `weight_sync.py` — NCCL broadcast + `engine.update_weights()` call.
- `loop.py` — the orchestrator that runs steps 1-6.
- `architecture.md` — this file.

Convergence is not the deliverable. A clean diagram + working sync + a non-trivial reward curve over a few hundred steps is.
