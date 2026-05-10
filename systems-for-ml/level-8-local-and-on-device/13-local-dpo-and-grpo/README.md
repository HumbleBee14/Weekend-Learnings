# 13 — Local DPO and GRPO

## Files

- `CONCEPTS.md` — DPO/ORPO/GRPO mechanics, why they fit on Mac, the 4-bit frozen reference trick, recipes, evaluation, pitfalls.
- `dpo_train.sh` — wrapper around `mlx_lm_lora.dpo` with sane defaults.
- `make_pref_dataset.py` — turns prompt + two-completion pairs into the JSONL preference format.
- `grpo_reward.py` — example reward function for GRPO: scores math answers via regex extraction.

## Quickstart

```bash
pip install mlx-lm-lora mlx-tune

# Build a tiny preference dataset (or bring your own).
python make_pref_dataset.py --output prefs.jsonl

# DPO from an SFT checkpoint (Topic 12 produced ./adapters; fuse first).
python -m mlx_lm.fuse \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --adapter-path ./adapters \
    --save-path ./sft-checkpoint

chmod +x dpo_train.sh
./dpo_train.sh
```

## Expected output

```
iter 10  loss=0.640  reward_margin=0.12
iter 100 loss=0.58   reward_margin=0.34
...
val reward_margin=0.42  (chosen logprob - rejected logprob, +ve = learning preferences)
```

`reward_margin` rising means the policy is increasingly favoring `chosen` over `rejected`. If it stalls at 0, your data is too easy (already-preferred) or too noisy.

## Try

- Run ORPO instead of DPO (`python -m mlx_lm_lora.orpo`). No reference model — should fit a couple GB tighter.
- GRPO smoke test: `python -m mlx_tune.grpo --model ./sft-checkpoint --prompts math_prompts.jsonl --reward ./grpo_reward.py --group-size 8 --iters 200`. Watch reward mean climb as the model learns to land the right number.
- Compare three checkpoints (base / SFT / DPO) on a 100-prompt LLM-as-judge head-to-head. Win rate is the thing you actually want to move.
- After DPO, re-run the **MMLU subset** check (Topic 12). DPO can shift general behavior — confirm it didn't blow up.

## Where this goes

This is the last training topic. The remainder of the level (Topics 14–16) is runtime: CPU paths, multi-Mac inference, privacy. Project 4's `local-agent` ships an SFT+DPO/ORPO model in the chat slot. **G20 of Project 4** uses the before/after numbers from this topic and Topic 12 together.
