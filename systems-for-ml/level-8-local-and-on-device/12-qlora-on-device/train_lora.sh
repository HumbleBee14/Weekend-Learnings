#!/usr/bin/env bash
# QLoRA training on M-series via mlx_lm.
#
# Defaults tuned for a 7B 4-bit base on 32-64 GB Mac.
# Adjust BATCH and SEQ if you OOM.

set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
DATA="${DATA:-./data}"
OUT="${OUT:-./adapters}"

BATCH="${BATCH:-4}"
SEQ="${SEQ:-2048}"
ITERS="${ITERS:-1000}"
RANK="${RANK:-16}"
LAYERS="${LAYERS:-16}"
LR="${LR:-1e-4}"

mkdir -p "$OUT"

python -m mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$DATA" \
    --batch-size "$BATCH" \
    --max-seq-length "$SEQ" \
    --iters "$ITERS" \
    --lora-layers "$LAYERS" \
    --lora-rank "$RANK" \
    --learning-rate "$LR" \
    --val-batches 25 \
    --steps-per-eval 100 \
    --steps-per-report 10 \
    --adapter-path "$OUT"

echo
echo "Adapter saved at $OUT"
echo "Generate with:"
echo "  python -m mlx_lm.generate --model $MODEL \\"
echo "      --adapter-path $OUT --prompt '...'"
