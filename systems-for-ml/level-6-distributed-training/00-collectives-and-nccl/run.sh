#!/usr/bin/env bash
# Two-GPU NCCL demo with INFO-level topology output.
set -e
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET
torchrun --standalone --nproc_per_node=2 collectives_demo.py
