#!/usr/bin/env bash
# Start two OpenAI-compatible local servers side-by-side for benchmarking.
#
# 1. Ollama with the MLX backend on :11434.
# 2. mlx_lm.server on :8000.
#
# Usage:
#   chmod +x start_servers.sh
#   ./start_servers.sh
#
# Stop with Ctrl-C. Each engine logs into ./logs/.

set -euo pipefail

mkdir -p logs

MODEL_OLLAMA="${MODEL_OLLAMA:-qwen2.5:7b-instruct-q4_K_M}"
MODEL_MLX="${MODEL_MLX:-mlx-community/Qwen2.5-7B-Instruct-4bit}"

echo "Starting Ollama (MLX backend) on :11434..."
OLLAMA_BACKEND=mlx ollama serve > logs/ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 2
ollama pull "$MODEL_OLLAMA" || true

echo "Starting mlx_lm.server on :8000..."
python -m mlx_lm.server \
    --model "$MODEL_MLX" \
    --port 8000 \
    > logs/mlx_lm.log 2>&1 &
MLX_PID=$!

echo
echo "Servers running."
echo "  Ollama  : http://localhost:11434/v1   pid=$OLLAMA_PID  model=$MODEL_OLLAMA"
echo "  mlx_lm  : http://localhost:8000/v1    pid=$MLX_PID     model=$MODEL_MLX"
echo
echo "Bench example:"
echo "  python bench_serving.py --base-url http://localhost:11434/v1 \\"
echo "      --model $MODEL_OLLAMA --concurrency 8 --requests 32"
echo
echo "Ctrl-C to stop both."
trap "kill $OLLAMA_PID $MLX_PID 2>/dev/null || true" EXIT
wait
