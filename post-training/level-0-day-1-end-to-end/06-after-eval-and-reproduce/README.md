# 06 — After Eval & Reproduce

Re-run the **exact same eval**, now with the adapter. The delta is the whole lesson.

```bash
# cloud / GPU
python evaluate.py --adapter out/sft-lora --limit 100 --out reports/sft.json
# Mac
python evaluate_mlx.py --adapter out/mlx-adapters --limit 100
```

## Read the delta

Put the two reports side by side (`reports/base.json` vs `reports/sft.json`). You should see:

```
parse_rate       ~0.5–0.9   →   ~1.0
field_accuracy   ~0.2–0.6   →   ~0.9+
exact_match      ~0.05–0.3  →   ~0.7–0.9
```

That jump, from one method, on a metric you defined, is what "post-training worked" means. Not a vibe — a number.

## Reproduce on the other backend

You chose cloud **and** Mac. Whichever you did first, now do the other. Same `data/`, same `task.py`, same eval — only the trainer changed (`train_cloud.py` ↔ `mlx_lm.lora`). Confirming the result holds across backends proves the *method* did the work, not some quirk of one framework. This is also your first taste of Level 8-style "same lesson, different bandwidth budget" from the systems track.

## The honest caveat (previews Level 2 & 5)

You measured the **target skill**. You have *not* yet checked whether the model got worse at everything else — that's **catastrophic forgetting**, and measuring it (a general-capability regression check) is exactly what Level 2 and Level 5 add. For today, the target-skill win is enough.

## Done with Level 0

Save your two reports. You've trained a model end-to-end and moved a real metric — on two backends. Say the sentence from the [runbook](../README.md#what-you-can-now-say), then decide: keep going to Level 1 (the map, deep), or bank the confidence and come back.
