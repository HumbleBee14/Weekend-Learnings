# 05 — First SFT Run

Now teach it. This is supervised fine-tuning: minimize the loss of producing the gold JSON, given the prompt.

```bash
# cloud / GPU
python train_cloud.py                       # -> adapter in out/sft-lora

# Mac
mlx_lm.lora --model Qwen/Qwen3-0.6B --train --data data \
    --iters 400 --batch-size 4 --num-layers 8 --adapter-path out/mlx-adapters
```

## What's actually happening (the 60-second version)

- **LoRA**, not full fine-tuning: the 0.6B base weights stay **frozen**; you train a few small adapter matrices bolted onto each layer. That's why it fits in little memory and trains fast — and why later methods (DPO, GRPO) can reuse the same trick. (The *why* of low-rank adapters is Level 2.)
- **Loss on the completion only**: the model is scored on the JSON, not the instruction.
- **`eos_token="<|im_end|>"`**: Qwen-specific — this teaches the model to *stop* after the JSON instead of rambling.
- Watch the **loss fall** in the logs. A falling loss = the model is learning to reproduce the gold completions. That's the entire visible signal of SFT.

## Knobs you'll understand deeply in Level 2

`r` / `lora_alpha` (adapter capacity), `learning_rate` (~1e-4 for adapters), `num_train_epochs`, `--num-layers` (MLX: how many layers get adapters). Defaults here are tuned to just work; don't fiddle yet.

When it finishes you'll have a saved adapter (`out/sft-lora` or `out/mlx-adapters`) — a few MB of learned deltas, not a new full model.

Next → [06 — after eval & reproduce](../06-after-eval-and-reproduce/).
