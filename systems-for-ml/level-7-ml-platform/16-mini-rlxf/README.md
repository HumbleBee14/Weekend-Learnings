# 16 — mini-RLXF

## Files

- `CONCEPTS.md` — RLXF as a platform problem, why rollouts run on vLLM, NCCL weight sync, on-policy vs off-policy, reward sources, how Level 7's other topics show up in the loop.
- `architecture.md` — the six-step loop diagram + per-file deliverables for `mini-platform/rlxf/`.
- `loop.py` — orchestration sketch. Real rollouts via vLLM's OpenAI API; reward / trainer / weight-sync stubs marked.

## Quickstart

```bash
# 1. vLLM serving a small base model (use whatever fits your GPU):
vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 8000

# 2. Run the loop sketch:
python loop.py
```

## Expected output

```
{"step_reward": 0.3333, "n": 24, "step": 0, "wall_s": 5.42}
{"step_reward": 0.3750, "n": 24, "step": 1, "wall_s": 5.10}
{"step_reward": 0.4167, "n": 24, "step": 2, "wall_s": 5.23}
{"step_reward": 0.4583, "n": 24, "step": 3, "wall_s": 5.04}
[step 4] NCCL broadcast + vLLM engine.update_weights()
...
```

The trainer step is a stub; reward will not actually rise without a real trainer behind it. The point is the *loop architecture* — not convergence.

## Try

- **Wire a real trainer.** Replace `trainer_step` with a call into your Level 6 FSDP2 trainer process. The trainer accepts (rollouts, advantages), runs one PPO/GRPO step, returns metrics.
- **Real weight sync.** Implement `weight_sync.py` using `torch.distributed.broadcast` to push weights into vLLM's HBM, then call vLLM's `update_weights` API. Time the sync; aim for sub-second on small models.
- **Reward variants.** Swap `reward_for` with a rule-grader for GSM8K, then with a reward-model call (a separate vLLM serving an RM), then with an LLM-as-judge.
- **Rollout pool autoscaling.** Run multiple vLLM rollout replicas behind your Topic 06 router. KEDA-scale the rollout pool on its own queue depth — separate from interactive traffic.
- **Off-policy.** Replace GRPO group-norm with DPO. Verify that without weight sync, the loop still trains (DPO is off-policy on a fixed preference corpus).

## Where this goes

- This is the closing piece of `mini-platform`. By the time it works end-to-end, you have a system that trains, evaluates, registers, routes, autoscales, observes, and rolls out RL improvements — every box in Topic 01's architecture.

## References

- verl — https://github.com/volcengine/verl
- OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
- NeMo-RL — https://github.com/NVIDIA/NeMo-RL
- DeepSeek-R1 / GRPO — https://arxiv.org/abs/2501.12948
- vLLM weight update — https://docs.vllm.ai/en/latest/
