# Level 2 — SFT, Deep

> Track map: [`post-training/README.md`](../README.md) · Primary read: [RLHF Book](https://rlhfbook.com) (Instruction Tuning) + [TRL `SFTTrainer` docs](https://huggingface.co/docs/trl)
>
> Goal: understand *exactly* what supervised fine-tuning changes in the model — token by token — and control it well enough to avoid the classic failure (a model that learns the task and forgets everything else).

## The WHY

SFT is **imitation learning**: show the model `prompt → ideal completion` pairs and train it to reproduce the completion. It's the cheapest, most reliable way to install a behavior — and the foundation every preference/RL method builds on. But the details that tutorials skip (what loss is computed on which tokens, how the chat template is applied, what packing does) are exactly the details that decide whether your fine-tune works.

## Where this fits

- **Comes after:** Level 0 (you ran SFT) + Level 1 (you can place it on the map).
- **Comes before:** Level 3 — DPO starts from your SFT checkpoint.

## Topics

| # | Topic | What you learn / build |
|---|-------|------------------------|
| 01 | next-token-loss-revisited | What cross-entropy on curated completions actually optimizes; why SFT ≠ pretraining despite the same loss |
| 02 | chat-templates-and-formatting | Special tokens, roles, why a wrong template silently wrecks a run |
| 03 | loss-masking | Compute loss on *completion* tokens only, not the prompt — and why it matters |
| 04 | packing-and-efficiency | Sequence packing, padding waste, throughput on small GPUs |
| 05 | lora-qlora-from-first-principles | *Why* low-rank adapters work; rank/alpha/target-modules; QLoRA's 4-bit base |
| 06 | catastrophic-forgetting | The general-capability regression; how much SFT is too much |
| 07 | sft-run-and-eval | The "real" SFT run on the JSON task + task/regression eval delta |

## What "done" looks like

- A properly-templated, loss-masked SFT run that beats Level 0's quick version on `field_accuracy`.
- A regression measurement showing the model *didn't* forget general ability (or a documented account of where it started to).
- You can explain rank/alpha and why QLoRA fits a 7B fine-tune in ~8 GB.

## Eval checkpoint

Run the full stack: **task eval** (parse-rate + field-F1) *and* the **regression slice** (MMLU-Pro / IFEval). Record the delta vs Level 0. This is the SFT baseline that DPO and GRPO must beat.
