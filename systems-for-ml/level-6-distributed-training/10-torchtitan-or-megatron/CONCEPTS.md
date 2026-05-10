# 10 — torchtitan or Megatron

This is where the week converges into a real artifact: a small model trained end-to-end with all the parallelism plumbing. The trained checkpoint is what Level 7's `mini-platform` will serve.

## Pick one

**torchtitan** ([github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)) — PyTorch-native, simpler install, ICLR 2025 paper. Recipes for Llama 3 (8B–405B), Llama 4, DeepSeek-V3-style MoE. **Recommended for this curriculum** — closer to what most non-NVIDIA labs use.

**Megatron-Core / Megatron-Bridge** ([github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [github.com/NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)) — NVIDIA-native, more parallelism axes, Transformer Engine integration, day-0 model support. Megatron-Bridge does bidirectional HuggingFace ↔ Megatron checkpoint conversion (the missing piece for years).

Both work. torchtitan is faster to bring up at small scale. Megatron-Core is what frontier NVIDIA-stack runs use.

## torchtitan in 2026

- Configuration via TOML.
- 5D parallelism (DP+TP+PP+EP+CP) wired through DeviceMesh.
- Async DCP checkpointing.
- Float8 (FP8) training via `torchao` integration.
- Async-TP for compute-comm overlap.
- ZeroBubble pipeline schedule.
- Mosaic StreamingDataset integration.

The README has Llama 3 8B and 70B configs that you can scale down.

## Megatron-Core in 2026

- Same axes plus Dynamic Context Parallelism (Jan 2026).
- Transformer Engine for FP8 + FP4 (Blackwell).
- MoE token dispatcher with no-token-dropping default.
- Heterogeneous-pipeline support (different stage sizes per device class).

Steeper install (TE, Apex, Megatron-LM). Worth it for production-scale runs.

## What you build this week

A small model — 100M to 1B params — trained for ~200 steps on 2 GPUs with FSDP2 (and optionally TP=2). The checkpoint goes to Level 7.

Choose a small base config:
- torchtitan `train_configs/llama3_8b.toml` → scale `n_layers` to 6, `dim` to 1024, `vocab_size` to 32K.
- Or write your own minimal config — see `mini_titan_config.toml` in this folder.

## Build steps

1. Clone torchtitan: `git clone https://github.com/pytorch/torchtitan && cd torchtitan && pip install -e .`
2. Use the `mini_titan_config.toml` here (or scale down a stock one).
3. Provide a small dataset. Easiest: torchtitan's built-in `c4_test` or a 1B-token TinyStories MDS.
4. `torchrun --standalone --nproc_per_node=2 -m torchtitan.train --job.config_file mini_titan_config.toml`
5. Watch the loss curve. Profile with PyTorch Profiler. Check that `nccl:all_gather` and `nccl:reduce_scatter` overlap with compute.
6. **Save the checkpoint.** It goes to Level 7.

## What you should profile

- **MFU**: roughly `tokens/sec × params × 6 / peak_flops`. Target 30–50% on small models, 50–70% on big ones at the right interconnect.
- **Loader vs model throughput**: see Topic 03. Confirm loader keeps ahead.
- **Comms vs compute**: in the profiler, sum the `nccl:*` time. If >25% of step time isn't overlapping with compute, your axis composition is wrong (likely TP across nodes, or FSDP without enough work between the gather and release).
- **Memory headroom**: peak/total. If <10% headroom you'll OOM under load (longer sequences, gradient spikes).

## Reference

- torchtitan repo: [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- torchtitan paper: [arxiv.org/abs/2410.06511](https://arxiv.org/abs/2410.06511)
- torchtitan blog: [pytorch.org/blog/training-with-torchtitan](https://pytorch.org/blog/training-with-torchtitan/)
- Megatron-LM: [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Megatron-Bridge: [github.com/NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- Megatron-Core docs: [docs.nvidia.com/megatron-core](https://docs.nvidia.com/megatron-core)
