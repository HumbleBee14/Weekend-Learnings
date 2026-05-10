#!/usr/bin/env bash
# Run lm-evaluation-harness on a quantized model and its BF16 reference.
#
# Usage:
#   pip install lm-eval vllm
#   chmod +x run_lm_eval.sh
#   ./run_lm_eval.sh
#
# Outputs results to ./eval_results/

set -euo pipefail

REFERENCE="Qwen/Qwen2.5-1.5B-Instruct"
QUANTIZED_DIR="${1:-./Qwen2.5-1.5B-Instruct-FP8}"
TASKS="mmlu,gsm8k,arc_easy,hellaswag"
RESULTS_DIR="eval_results"

mkdir -p "$RESULTS_DIR"

echo "=== BF16 reference ==="
lm-eval \
    --model hf \
    --model_args "pretrained=$REFERENCE,dtype=bfloat16" \
    --tasks "$TASKS" \
    --batch_size 8 \
    --output_path "$RESULTS_DIR/bf16"

echo ""
echo "=== Quantized: $QUANTIZED_DIR ==="
lm-eval \
    --model vllm \
    --model_args "pretrained=$QUANTIZED_DIR,gpu_memory_utilization=0.8" \
    --tasks "$TASKS" \
    --batch_size 8 \
    --output_path "$RESULTS_DIR/quantized"

echo ""
echo "Compare scores in $RESULTS_DIR/bf16/results.json and $RESULTS_DIR/quantized/results.json"
echo "Tolerance: <1% drop on MMLU, similar on others, before declaring the recipe production-ready."
