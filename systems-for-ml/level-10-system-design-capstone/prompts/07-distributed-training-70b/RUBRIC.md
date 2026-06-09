# Prompt 07 — Rubric

## Strong signals
- Correct parallelism *dimensionality* for the workload: 2D (FSDP×TP) suffices for a dense 70B on 256 GPUs; 3D (add PP) at 10× scale. Knows "5D" (adding context + expert parallelism) is for long-context / MoE, not a dense 70B. Explicit TP/DP/PP sizing math.
- **Goodput**, not just MFU — names failure budget, recovery time
- **Async DCP** checkpointing + peer replication
- Data pipeline as a first-class concern (tokenizer pool, sequence packing)
- NCCL Communicator Shrink for elastic recovery
- Names torchtitan or Megatron-Core specifically

## Anti-signals
- "We'd use DDP" — that's data-parallel only, won't fit 70B
- DeepSpeed ZeRO-3 as the primary answer — legacy; FSDP2 is current
- Synchronous checkpointing that stalls training
- No data-pipeline story — GPUs will starve at this scale
- One-rank-dies → restart whole world from scratch

## What's tested
Full Level 6 integration. The training-side cousin of Prompt 01.
