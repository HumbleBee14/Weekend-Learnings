# 04 — Baseline Eval (watch it fail)

Measure the model **before** you touch it. You can't claim an improvement without a baseline.

```bash
# cloud / GPU
python evaluate.py --limit 100 --out reports/base.json
# Mac
python evaluate_mlx.py --limit 100
```

## What you're looking at

The script runs the base model on 100 held-out records and prints:

- `parse_rate` — fraction of outputs that are valid JSON at all
- `field_accuracy` — fraction of the 5 fields that match ground truth
- `exact_match_rate` — fraction where *all* fields are right

A 0.6B base model typically wraps answers in prose or code fences, invents keys, or fumbles a field — so expect a **low, telling** baseline. Read the `first few outputs` the script prints; *seeing* the failure modes is the point.

## Why this is the same object as the reward

`evaluate.py` calls `score()` from `task.py`. In Level 4 the **exact same `score()`** becomes the GRPO reward function. Eval and reward are one verifiable checker — so a good baseline eval today is also the foundation of RL later.

Save `reports/base.json`. You'll diff it against the post-SFT run.

Next → [05 — first SFT run](../05-first-sft-run/).
