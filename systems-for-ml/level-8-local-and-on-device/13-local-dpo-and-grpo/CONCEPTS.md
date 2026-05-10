# 13 — Local DPO and GRPO

Preference learning without a full RLHF stack. Both methods skip the explicit reward model. DPO needs a reference model; GRPO needs only a reward function. Both are tractable on a single Mac in 2026.

## Why this matters locally

Classical RLHF needs a policy, a reference model, a reward model, often a value head, plus PPO's machinery. Three 7B models in fp16 plus PPO state on a laptop is not happening. DPO and GRPO collapse this:

```
  RLHF (PPO):   policy + ref + reward + value  (4 models, ~150 GB RAM at fp16)
  DPO:          policy + ref                    (2 models; ref can be 4-bit frozen)
  GRPO:         policy + reward function        (1 model + a Python function)
```

This is why most local 2026 practitioners do SFT (Topic 12) -> DPO/ORPO/GRPO -> ship. Skip PPO.

## DPO — direct preference optimization

### The loss

Given preference pairs `(prompt, chosen, rejected)`:

```
  L_DPO = -log sigmoid( beta * [log pi_theta(chosen)/pi_ref(chosen)
                              - log pi_theta(rejected)/pi_ref(rejected)] )
```

Plain English: increase the log-prob ratio of `chosen` over `rejected` under your policy, *relative to* the reference. The `pi_ref` keeps you from drifting arbitrarily far from the SFT base. `beta` controls how strongly the reference anchors you (typical 0.1).

### The trick that makes it fit on Mac

Run reference model **frozen and 4-bit**, while policy is fp16 with LoRA on top. `mlx-lm-lora` (canonical) does this by default; `mlx-tune` is a smaller third-party alternative with similar mechanics. So in practice:

```
  ref (4-bit, frozen)     ~  5 GB
  policy base (4-bit)     ~  5 GB  (shared with ref structurally — same weights)
  LoRA adapters (fp16)    ~ 30 MB
  optimizer state         ~ 60 MB
                          ----------
                          ~ 10 GB-ish
```

A 7B DPO run is comfortable on 32 GB Macs.

### Recipe

```bash
pip install mlx-lm-lora

python -m mlx_lm_lora.dpo \
    --model ./sft-checkpoint \
    --train \
    --data ./prefs.jsonl \
    --beta 0.1 \
    --learning-rate 5e-7 \
    --iters 500 \
    --lora-rank 16
```

Each line of `prefs.jsonl`:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

500–2000 preference pairs is enough to move the model meaningfully. LR is unusually low (5e-7) — DPO is sensitive.

## ORPO — odds-ratio preference optimization

Even simpler: combines SFT loss and a preference-based odds ratio in a single step. No reference model. Quality competitive with DPO on 7B-scale models. Worth trying as the lightest option.

```
  L_ORPO = L_SFT(chosen) - lambda * log_odds_ratio(chosen vs rejected)
```

`mlx-lm-lora` supports it; switch the entry point to `orpo` with the same data shape.

## GRPO — group relative policy optimization

DeepSeek's contribution. The big 2026 win for local because:

- **No reference model.** Saves ~5 GB.
- **No value head.** Saves another optimizer state.
- **Reward is a Python function.** Can be deterministic (test-passes / regex / format check) or learned. For coding tasks this is gold — the reward is "did the unit tests pass."

### The mechanic

For each prompt, sample G completions (group size, e.g., 8). Compute reward for each. Normalize within the group:

```
  A_i = (r_i - mean(r)) / std(r)
```

Update policy to increase log-prob of high-A completions, decrease low-A. KL penalty against the SFT model keeps things stable.

```
  +-----------+
  |  prompt   |
  +-----------+
        |
        v
  sample 8 completions, score each
        |
        v
  +-------+-------+-------+-------+-------+-------+-------+-------+
  | r=0.9 | r=0.8 | r=0.7 | r=0.5 | r=0.3 | r=0.2 | r=0.1 | r=0.0 |
  +-------+-------+-------+-------+-------+-------+-------+-------+
        |  normalize, gradient up on top half, down on bottom half
        v
   updated policy
```

### Why GRPO fits Mac so well

Sampling G completions is cheap on Apple Silicon thanks to unified memory and lazy MLX scheduling. You can keep a single 7B in memory and roll out groups of 8 within the same address space. No per-sampler GPU launch overhead from CUDA-style code.

### Recipe

```bash
pip install mlx-tune

python -m mlx_tune.grpo \
    --model ./sft-checkpoint \
    --train \
    --prompts ./prompts.jsonl \
    --reward ./reward.py \
    --group-size 8 \
    --iters 500 \
    --kl-coef 0.04
```

`reward.py` exports a function `score(prompt: str, completion: str) -> float`. For code: run unit tests, return pass rate. For math: regex extract + check. For chat: an LLM-as-judge call with an external API (acceptable for *training*, not for inference — same threat model conversation as Topic 16).

## What does not work locally

- PPO with simultaneous policy + ref + reward + value heads at 13B+ in fp16. Always OOMs on 64 GB.
- RLAIF where the AI judge is a 70B running locally during training — too slow. Use a smaller judge or batch the API calls.
- Online RL with megabatches. The Mac is not a 8xH100 cluster; keep groups small and accept slower wall time.

## Evaluation

Always measure:

1. **Reward mean on a held-out prompt set.** Trained reward should rise on training prompts and not collapse on held-out.
2. **KL to base.** If KL drifts huge (> 30 nats per response), you've over-trained — decrease LR or increase KL coefficient.
3. **General-benchmark slice.** Same forgetting check as Topic 12.
4. **Win rate vs base.** LLM-as-judge head-to-head on 100 prompts.

## Common pitfalls

1. **DPO LR too high.** 5e-5 is the LoRA SFT LR; for DPO use 5e-7. Off by 100×.
2. **Bad preference pairs.** If `chosen` and `rejected` differ in length or formatting only, the model learns formatting not preference. Match style; differ on substance.
3. **GRPO with a noisy reward.** Sparse rewards (0/1) plus group size 4 = high variance. Use group size 8–16 and a denser reward where you can.
4. **Skipping the SFT step.** DPO/GRPO from a base model usually fails — the initialization is too far from preferred behavior. SFT first.
5. **No KL clamp.** GRPO can collapse to deterministic outputs; KL coefficient (0.02–0.1) holds entropy.

## References

- DPO paper: https://arxiv.org/abs/2305.18290
- ORPO paper: https://arxiv.org/abs/2403.07691
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1 (GRPO at scale): https://arxiv.org/abs/2501.12948
- mlx-lm-lora: https://github.com/Goekdeniz-Guelmez/mlx-lm-lora
- mlx-tune: https://github.com/ARahim3/mlx-tune
- TRL (reference for the same algorithms in Python torch): https://github.com/huggingface/trl
