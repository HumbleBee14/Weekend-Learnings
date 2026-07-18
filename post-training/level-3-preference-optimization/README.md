# Level 3 — Preference Optimization (Offline)

> Track map: [`post-training/README.md`](../README.md) · Primary read: [DPO paper](https://arxiv.org/abs/2305.18290) (Rafailov 2023) + [RLHF Book](https://rlhfbook.com) (DPO / preference chapters)
>
> Goal: understand why the field detoured through reward models and RLHF, then why **DPO** collapsed that whole pipeline into a single loss — and run DPO on your SFT checkpoint.

## The WHY

SFT teaches the model *a* good answer. But "good" is often comparative — answer A is better than answer B — and there's no single target to imitate. The original solution (RLHF) trained a **reward model** on human preferences, then used **PPO** to optimize against it: powerful, but three models and a fragile RL loop. **DPO's insight:** the optimal RLHF policy has a closed-form relationship to the reward, so you can skip the reward model *and* the RL loop and optimize preferences with a single supervised-style loss. This is the offline preference row of the map.

## Where this fits

- **Comes after:** Level 2 — DPO fine-tunes the SFT model, not the base.
- **Comes before:** Level 4 — where we add live generation (online) and true RL.

## Topics

| # | Topic | What you learn / build |
|---|-------|------------------------|
| 01 | why-rlhf-existed | The reward-model → PPO pipeline; what problem it solved and what it cost |
| 02 | dpo-the-insight | Bradley-Terry model, the implicit reward, the DPO loss — derived intuitively |
| 03 | reference-model-and-kl | Why DPO keeps a frozen reference model; the KL leash that stops drift |
| 04 | building-preference-data | Turning the JSON task into `(chosen, rejected)` pairs; on-policy vs off-policy pairs |
| 05 | dpo-variants | KTO (no pairs needed), ORPO (no reference model), IPO — when each wins |
| 06 | dpo-run-and-eval | DPO on the SFT checkpoint + task/regression eval delta |

## What "done" looks like

- A preference dataset built from your task (valid+correct = chosen, broken/wrong = rejected).
- A DPO run that measurably tightens the model vs the SFT baseline.
- You can explain, without notes, why DPO needs a reference model and what the β/KL term controls.

## Eval checkpoint

Full stack again: task + regression delta **vs the Level 2 SFT model**. Note whether DPO helped the metric SFT couldn't — and whether it cost any general ability.
