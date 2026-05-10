#!/usr/bin/env bash
# DPO training on M-series via mlx-lm-lora.
#
# Reference model is the same SFT checkpoint, loaded frozen 4-bit.
# Policy adds a fp16 LoRA on top.

set -euo pipefail

MODEL="${MODEL:-./sft-checkpoint}"
DATA="${DATA:-./prefs.jsonl}"
OUT="${OUT:-./dpo-adapters}"

BETA="${BETA:-0.1}"
LR="${LR:-5e-7}"
ITERS="${ITERS:-500}"
RANK="${RANK:-16}"

mkdir -p "$OUT"

python -m mlx_lm_lora.dpo \
    --model "$MODEL" \
    --train \
    --data "$DATA" \
    --beta "$BETA" \
    --learning-rate "$LR" \
    --iters "$ITERS" \
    --lora-rank "$RANK" \
    --batch-size 1 \
    --max-seq-length 1024 \
    --steps-per-eval 50 \
    --steps-per-report 10 \
    --adapter-path "$OUT"

echo
echo "DPO adapter saved at $OUT"
