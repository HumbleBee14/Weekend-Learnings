#!/usr/bin/env bash
# Sweep straggler severity. Captures the curve for G11.
set -e
for ms in 0 5 10 25 50 100 200; do
    echo "===== slow_ms=$ms ====="
    torchrun --standalone --nproc_per_node=2 straggler_inject.py --slow_ms $ms
done
