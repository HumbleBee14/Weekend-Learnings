# 12 — QLoRA On-Device

## What QLoRA actually does

```
   +-------------------------+
   |  Base model weights     |   <-- frozen, 4-bit quantized
   |  ~5 GB for a 7B at 4bit |
   +-------------------------+
                +
   +-------------------------+
   |  LoRA adapters          |   <-- trainable, fp16
   |  ~30 MB at rank 16      |   <-- about 0.5% of base
   +-------------------------+
                =
   +-------------------------+
   |  Specialized model      |
   +-------------------------+
```

The trick: **only the adapters update during training**. Gradients flow through the quantized base into the adapters, but the base never changes. Memory cost is dominated by the (frozen, 4-bit) base plus optimizer state on a tiny adapter, not on a fp16 7B with Adam moments.

For Mac, this is what makes 7B fine-tuning practical on a 32 GB laptop and 70B fine-tuning practical on an M5 Max 128 GB.

## What MLX-LM gives you in 2026

`mlx_lm.lora` is the canonical entry point. Single CLI:

```bash
python -m mlx_lm.lora \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --train \
    --data ./data \
    --iters 1000 \
    --batch-size 4 \
    --lora-layers 16 \
    --lora-rank 16 \
    --learning-rate 1e-4
```

If the base is already 4-bit, MLX-LM auto-detects it and runs QLoRA. If the base is fp16, it runs vanilla LoRA. There is no separate flag.

`./data` expects two files: `train.jsonl` and `valid.jsonl`. Each line is one of:

```json
{"text": "raw text"}
{"prompt": "...", "completion": "..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Practical numbers (2026)

| Hardware | Model | Batch | Seq | tok/s training | Notes |
|---|---|---|---|---|---|
| M2 Max 32 GB | Mistral-7B 4-bit | 2 | 1024 | ~700 | Comfortable |
| M3 Max 64 GB | Qwen2.5-7B 4-bit | 4 | 2048 | ~1500 | Sweet spot |
| M3 Max 64 GB | Llama-3.1-13B 4-bit | 2 | 1024 | ~600 | Tight; close other apps |
| M5 Max 128 GB | Llama-3.1-70B 4-bit | 1 | 1024 | ~120 | Slow but real |
| M5 Max 128 GB | Qwen2.5-7B 4-bit | 8 | 2048 | ~3500 | M5 NA win |

5k examples of an SFT dataset on a 7B at batch 4 / seq 2048 finishes in ~60–90 minutes on M3 Max.

## The data step is what determines quality

A useful local fine-tune uses *your* data. Three useful kinds:

1. **Style transfer.** A few hundred of your past notes / commits / emails. The model picks up cadence and vocabulary fast.
2. **Domain extraction.** Pairs of (raw doc, structured JSON) from your codebase or knowledge base. Trains schema adherence harder than prompting can.
3. **Tool-use bootstrapping.** Synthetic traces of correct tool calls for your specific tools (Topic 11). Massively reduces malformed-call rate from a small base.

Aim for 500–5000 examples for a usable LoRA. Below 200, the adapter mostly memorizes; above 50000, you should ask whether SFT is the right tool (probably not — that's pretraining-shaped).

## Catastrophic forgetting

A LoRA that nails your style and now fails MMLU is a regression. Always run a **before/after** general-benchmark slice. Even 200 MMLU questions catch the worst regressions.

```
  before LoRA: MMLU subset 0.62
  after  LoRA: MMLU subset 0.55  <-- 7 pt drift, too much
                                     reduce rank, fewer iters, or mix in
                                     general data
```

Mitigations that actually help:

- **Smaller rank** (8 instead of 32). Less capacity to overwrite.
- **Mix general data.** 70% your data + 30% something like UltraChat. Keeps the model's general behavior intact.
- **Fewer iterations.** Watch validation loss; stop when it plateaus on your validation set, not when it reaches zero on training set.
- **Higher rank, lower LR** (counterintuitive — but lower LR over more layers can be gentler than concentrated low-rank surgery).

## Fusing for serving

After training, you have an adapter at `./adapters/lora.safetensors`. Two options to serve:

```bash
# Option A: load adapter alongside base at inference time
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --adapter-path ./adapters \
    --prompt "..."

# Option B: fuse into a new base for faster load and smaller footprint
python -m mlx_lm.fuse \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --adapter-path ./adapters \
    --save-path ./qwen-7b-mystyle
```

Fused models load faster but cost disk space. For one specialized model, fuse. For a model server that swaps between many adapters, keep them separate and pass `--adapter-path` per request.

## DoRA, LoRA+, and rank-stabilized variants

`mlx_lm.lora` supports a few 2025/2026 variants:

- `--use-dora` — DoRA (weight-decomposed LoRA). Better quality at the same rank, ~10% slower training.
- LoRA+ — separate LR for `A` and `B` matrices. Modest quality bump.
- Rank-stabilized scaling (`alpha = sqrt(rank)`). Default in recent mlx-lm.

DoRA is the default-on choice in 2026 for any new fine-tune unless you're benchmarking against a specific baseline.

## Common pitfalls

1. **No validation set.** You'll overtrain. Always carve out 5–10%.
2. **Eval on train data.** Same.
3. **Forgetting catastrophic-forgetting check.** 200 MMLU questions before/after, every time.
4. **Rank too high on small data.** Rank 64 on 300 examples = pure memorization.
5. **Wrong chat template.** Saving training data without the model's actual chat template applied means the LoRA learns the wrong start/end markers. Use `mlx_lm.utils.apply_chat_template` or generate data with the official tokenizer.
6. **Fusing too early.** Keep the adapter separate during exploration; fuse for production.

## References

- mlx-lm LoRA docs: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- QLoRA paper: https://arxiv.org/abs/2305.14314
- DoRA paper: https://arxiv.org/abs/2402.09353
- LoRA paper: https://arxiv.org/abs/2106.09685
- Apple ML — fine-tuning on M-series: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- lm-eval-harness for the catastrophic-forgetting check: https://github.com/EleutherAI/lm-evaluation-harness
