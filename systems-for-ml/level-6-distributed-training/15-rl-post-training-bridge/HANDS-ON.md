# Hands-on — Touching the Two-Engine Pattern

> CONCEPTS.md tells you the architecture; this file gets you to run it. The point is **not** to do real RLHF (that's its own curriculum). The point is to feel the trainer↔rollout-engine handoff in working code, so the architecture stops being abstract.

This is a **smoke test, not a research run.** ~30 minutes from clone to first rollout. The bottleneck you'll observe (rollout speed dominates training time) is the entire reason this topic exists.

## What you're going to build

A toy GRPO-style loop:

```
   ┌─────────────┐    prompts (8 in flight)    ┌──────────────┐
   │   trainer   │ ──────────────────────────► │ vLLM rollout │
   │ (TRL/GRPO)  │ ◄────── completions ─────── │   engine     │
   │ holds θ_train│                             │ holds θ_serve│
   └─────────────┘                              └──────────────┘
         │
         │ every 4 steps: weight sync (θ_train → θ_serve)
         ▼
       repeat
```

A tiny base model (Qwen2.5-0.5B or TinyLlama-1.1B), 1 GPU, a reward function that's just *"longer responses get higher reward, but penalize repetition"* — silly but it produces real gradients.

## Quickstart

```bash
# Requires: 1× GPU with ≥12GB (Colab T4, RTX 4090, L4, etc.)

pip install "trl>=0.13" "vllm>=0.6" "transformers>=4.46" "datasets" "accelerate"

python smoke_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --rollout-engine vllm \
  --batch-size 8 \
  --num-generations 4 \
  --steps 20
```

## Expected output shape

```
[init]   loading Qwen/Qwen2.5-0.5B-Instruct on cuda:0
[init]   starting vLLM rollout engine on cuda:0 (shared GPU, mem_fraction=0.4)
[step 0] rollout: 32 completions in 1.83s  (17.5 compl/s)
         reward: mean=0.42  std=0.21
         policy loss: 0.0034
[step 1] rollout: 32 completions in 1.81s
         reward: mean=0.45  std=0.19
         policy loss: 0.0029
...
[step 4] WEIGHT SYNC: trainer → vllm engine  (0.7s, full BF16 copy)
[step 5] rollout: 32 completions in 1.79s
         reward: mean=0.51  std=0.18   ← reward climbing as policy improves
...
[done]   20 steps, 12m 14s, avg compl/s 17.5
         time breakdown:
            rollout     76.4%
            training    18.2%
            weight_sync  5.4%
```

**The headline observation** — *rollout time dominates*. Even on this toy setup, the vLLM inference loop spends 4× more wall time than the gradient updates. This is universal in RL post-training and the entire reason `verl`, `OpenRLHF`, and `NeMo-RL` exist: to disaggregate the rollout pool from the trainer pool and scale them independently.

## What `smoke_grpo.py` actually does

You can either write it yourself (good exercise — TRL's `GRPOTrainer` does most of the heavy lifting) or use the reference at the bottom of this file. The structure:

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# 1. Tiny prompt dataset — a few dozen questions
prompts = ["What is 2+2?", "Capital of France?", ...]  # ~50 prompts
ds = Dataset.from_dict({"prompt": prompts})

# 2. Reward function — silly but real
def reward_fn(completions, **kwargs):
    rewards = []
    for c in completions:
        r = min(len(c.split()), 100) / 100.0   # length up to 100 words
        if "  " in c or c.count(c.split()[0]) > 5:  # repetition penalty
            r -= 0.3
        rewards.append(r)
    return rewards

# 3. GRPO config — TRL handles the vLLM rollout engine internally
cfg = GRPOConfig(
    output_dir="./out",
    num_generations=4,        # samples per prompt (the G in GRPO)
    per_device_train_batch_size=8,
    use_vllm=True,            # ← this is the switch
    vllm_gpu_memory_utilization=0.4,  # share 1 GPU with trainer
    max_steps=20,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=cfg,
    train_dataset=ds,
    processing_class=tokenizer,
)
trainer.train()
```

The thing to internalize: **`use_vllm=True` is where Level 5 walks into Level 6.** TRL is the trainer; vLLM is the rollout engine; the handoff between them is the bridge this whole topic is about.

## Variations to try (still in the smoke-test budget)

1. **Disable `use_vllm`** and rerun. Watch rollout time 5-10× and reward signal degrade because you can only afford fewer completions per step. This is *why* every production RLHF stack uses a dedicated inference engine.
2. **Increase `num_generations` from 4 → 16**. Watch rollout time climb linearly while training time stays roughly flat. This is the lever that disaggregated-rollout setups optimize for at scale.
3. **Swap `Qwen2.5-0.5B` → `TinyLlama-1.1B`**. Notice the rollout-vs-trainer ratio doesn't shift much — both scale with model size, the *ratio* is governed by inference vs training compute patterns.
4. **Plug in SGLang as the rollout engine** (if you have it installed). TRL doesn't natively support it yet; verl does. The fact that frameworks are still standardizing this interface tells you how new the architecture is.

## What this is **not**

- Not a real RL run. The reward function is a toy; 20 steps don't produce a useful model.
- Not multi-node. Real RLHF runs separate the rollout pool to a different cluster.
- Not multi-GPU per side. vLLM + trainer share one GPU here via `gpu_memory_utilization=0.4`.
- Not the place to debug reward hacking, KL divergence collapse, or off-policy correctness. Those belong in a dedicated RL curriculum.

If you finish this and want more, the next step is **verl**: it does the full disaggregated pattern with FSDP trainer + vLLM rollout on separate worker pools. Start with verl's `examples/grpo_trainer/` and the HybridFlow paper. But that's a separate weekend — don't blur the boundary.

## Where this goes

- Level 7's `mini-rlxf` (Topic 16) — the platform side. Same two engines, but the orchestration (job submission, checkpoint registry, eval gate before promoting a policy) is the focus.
- Level 5 Topic 11 (`offline-batch-inference`) — same shape: vLLM as a high-throughput generation backend that other systems feed prompts into. The only difference is whether the consumer is a trainer or a batch job.

## References

- [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [verl HybridFlow paper](https://arxiv.org/abs/2409.19256) — the canonical reference architecture
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) — GRPO at frontier scale
- [vLLM as RL rollout backend (docs)](https://docs.vllm.ai/en/latest/serving/integrations.html)
