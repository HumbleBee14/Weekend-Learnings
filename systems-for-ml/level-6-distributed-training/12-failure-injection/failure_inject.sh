#!/usr/bin/env bash
# Run elastic_train.py and kill one rank mid-training.
# Watch the surviving ranks log a NCCL watchdog timeout and the agent restart.
set -e

# Launch in background
torchrun \
    --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    --rdzv_id=fail_test --max-restarts=3 \
    elastic_train.py &

LAUNCH_PID=$!
echo "torchrun pid: $LAUNCH_PID"

sleep 8   # let it train for a bit

# Find one of the worker children and kill it
WORKER_PIDS=$(pgrep -P $LAUNCH_PID -f elastic_train)
KILL_PID=$(echo "$WORKER_PIDS" | head -1)
echo "killing worker pid $KILL_PID"
kill -9 "$KILL_PID"

# Let recovery happen
wait $LAUNCH_PID || true
echo "run finished (with restarts)"
