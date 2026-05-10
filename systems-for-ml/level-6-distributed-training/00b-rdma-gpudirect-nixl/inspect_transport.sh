#!/usr/bin/env bash
# Run a tiny NCCL job and grep for the transport NCCL chose.
# Reads NCCL_DEBUG=INFO log to identify NVL / P2P/IPC / SHM / NET/IB / NET/Socket.
set -e
LOG=$(mktemp)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
    torchrun --standalone --nproc_per_node=2 \
    ../00-collectives-and-nccl/collectives_demo.py 2> "$LOG" >/dev/null || true

echo "=== Transport per channel ==="
grep -E "Channel [0-9]+ : .* via " "$LOG" | sort -u | head -20
echo
echo "=== Plugin / network detection ==="
grep -E "NET/IB|NET/Socket|NET Plugin|GDRDMA" "$LOG" | sort -u | head -10
echo
echo "(Full log: $LOG)"
