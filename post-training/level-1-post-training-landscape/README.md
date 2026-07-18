# Level 1 — Landscape, Two Axes & the Three Pillars

> Track map: [`post-training/README.md`](../README.md) · Primary read: [RLHF Book](https://rlhfbook.com) ch. 1–3
>
> Goal: turn the pile of acronyms (SFT, DPO, RLOO, GRPO, PPO…) into a *map* you can place anything on, and understand why post-training exists at all.

## The WHY

Pretraining optimizes one objective: predict the next token over the whole internet. That produces a model that *knows* enormous amounts but only *continues text* — it doesn't reliably answer, refuse, follow a format, or prefer the better of two answers. **Post-training is how you install those behaviors** without paying to pretrain again. Level 1 is the conceptual spine; no heavy training here.

## Where this fits

- **Comes after:** Level 0 — you've already trained something, so this explains what you did.
- **Comes before:** Levels 2–4, which are one method-family each.

## Topics

| # | Topic | What you learn |
|---|-------|----------------|
| 01 | why-post-training | Pretraining vs post-training; what next-token prediction leaves undone; the alignment gap |
| 02 | the-two-axis-map | online/offline × imitation/preference/reward — the master diagram of the whole field |
| 03 | data-algorithm-co-design | Each method's *data contract*; why you can't pick an algorithm without picking a data shape |
| 04 | the-three-pillars | data⊗algorithm + reliable library + evaluation — and why eval is co-equal |
| 05 | eval-philosophy | The eval landscape (human / judge / static / agent), task vs regression, contamination & Goodhart |

## What "done" looks like

- You can draw the two-axis map from memory and place SFT, DPO, Online DPO, PPO, GRPO, RLVR on it.
- You can state, for any method, what data it consumes and what signal it uses.
- You can explain why measuring *only* the target skill is a trap (catastrophic forgetting).

## Eval checkpoint

No new training, but this is where the **eval discipline** is formalized: always measure *(1) did the target skill improve* and *(2) did anything else break*. Every later level obeys this.

## Teach-back

End the level by explaining the two-axis map out loud, from base model to GRPO, in under two minutes. If it stumbles, we re-run with a different framing.
