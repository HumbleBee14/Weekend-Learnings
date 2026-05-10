# 12 — QLoRA On-Device

## Files

- `CONCEPTS.md` — what QLoRA does, MLX-LM's flow, training numbers by hardware, the data step, catastrophic forgetting, DoRA / LoRA+ variants, fusing for serving.
- `train_lora.sh` — wrapper around `mlx_lm.lora` with the right defaults for a 7B QLoRA on M-series.
- `make_dataset.py` — turns a folder of plain-text / Markdown files into the `mlx_lm` JSONL format with the model's chat template applied.

## Quickstart

```bash
pip install mlx-lm

# 1. Build a small style-transfer dataset from your own writing.
python make_dataset.py \
    --input ./my-notes \
    --output ./data \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --val-frac 0.1

# 2. Train.
chmod +x train_lora.sh
./train_lora.sh

# 3. Generate with the adapter.
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --adapter-path ./adapters \
    --prompt "Write me a paragraph in my voice about unified memory."
```

## Expected output

`train_lora.sh` prints per-iter loss, validation loss every 100 iters, and saves checkpoints under `./adapters/`. On M3 Max 64 GB you should see ~1500 train tok/s and the validation loss drop from ~2.0 to ~1.3 over 1000 iterations on a few-thousand-example dataset.

## Try

- **Catastrophic-forgetting check (G20 of Project 4).** Run a 200-question MMLU subset with `lm-eval-harness` before and after the LoRA. If accuracy dropped > 3 points, reduce rank or iters.
- Add `--use-dora` to `train_lora.sh` and re-run. DoRA usually gains 1–2 points of quality at the same rank.
- Fuse: `python -m mlx_lm.fuse --model <base> --adapter-path ./adapters --save-path ./fused-model`. Compare load time and inference tok/s.
- Mix general data: 70% your notes + 30% UltraChat. Compare forgetting metrics — the mixed run should hold MMLU much better.

## Where this goes

Topic 13 layers DPO/GRPO on top of this SFT'd model — preference learning that respects what your style LoRA already taught it. The fused output of either is what you serve in the agent (Topic 11).
