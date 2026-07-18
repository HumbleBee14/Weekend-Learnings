# Level 4 — Online RL: PPO → GRPO → RLVR

> Track map: [`post-training/README.md`](../README.md) · Primary read: [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) + [DeepSeek-R1](https://arxiv.org/abs/2501.12948) + [RLHF Book](https://rlhfbook.com) (policy-gradient / RL chapters)
>
> Goal: understand the policy-gradient family from REINFORCE to GRPO, why "online" (the model learns from its *own* generations) matters, and run **GRPO with a verifiable reward** on the JSON task.

## The WHY

DPO learns from a *frozen* set of preference pairs — the model never sees how its *current* outputs are judged. **Online RL closes that loop:** the model generates answers, a reward function scores them *now*, and it updates from its own fresh mistakes. When the reward is **machine-verifiable** (does the JSON parse? does the math check?), you don't even need a reward model — that's **RLVR**, and it's the engine behind reasoning models like DeepSeek-R1. This is the bottom-right cell of the map, and the payoff of picking a verifiable task on Day 0.

## Where this fits

- **Comes after:** Level 3 — GRPO typically starts from an SFT (or DPO) checkpoint.
- **Comes before:** Level 5 — full-pipeline reproduction and honest eval.

## Topics

| # | Topic | What you learn / build |
|---|-------|------------------------|
| 01 | online-preference-onramp | Online DPO / XPO — the bridge: live generation, still a preference signal |
| 02 | policy-gradient-and-reinforce | REINFORCE: reward × log-prob; the root of every method in this row |
| 03 | rloo-and-baselines | Why variance kills naive REINFORCE; baselines; RLOO (leave-one-out, no critic) |
| 04 | ppo-and-its-pain | Clipping, the value/critic model, the reward model — power and cost |
| 05 | grpo | Group-relative advantage — PPO's quality without the critic; the R1 recipe |
| 06 | rlvr-and-reward-design | Verifiable rewards; wiring the Level-0 JSON checker in as the reward function |
| 07 | reward-hacking | How the model games the metric; shaping, format vs correctness, KL guardrails |
| 08 | grpo-run-and-eval | GRPO run + eval delta; the frontier named (DAPO, GSPO, Dr.GRPO) |

## What "done" looks like

- A GRPO run whose **reward function is literally your Level-0 eval verifier**.
- Evidence you looked for reward hacking (e.g., the model emitting trivially-valid but useless JSON) and how you countered it.
- You can explain the REINFORCE → PPO → GRPO progression and what each step added or removed.

## Eval checkpoint

Full stack vs the Level 3 model. This is where the "same task, three lenses" comparison completes: SFT vs DPO vs GRPO on one axis. Save all three — Level 5 writes them up.
