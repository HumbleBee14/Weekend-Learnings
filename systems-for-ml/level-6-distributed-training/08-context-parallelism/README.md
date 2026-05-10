# 08 — Context Parallelism

## Files

- `CONCEPTS.md` — Ring vs Striped Attention, when CP triggers, Dynamic CP (Megatron-Core Jan 2026)
- `cp_ring_demo.py` — minimal CP=2 ring with online-softmax accumulation; the rotation pattern is the part to study

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 cp_ring_demo.py
```

## Expected output

```
rank0: each device computed attention on its 1024 Q tokens against all 2048 K/V tokens via 2-step ring
rank0: output norm: 32.74
```

## Try

- Increase `world` to 4 (if you have 4 GPUs). Watch the K/V rotation visit every device.
- Add a causal mask. Without it the demo is non-causal attention. With it you'll see the load-imbalance problem ring attention has and striped fixes.
- Time the rotation. With long sequences, the rotation cost should overlap with the per-step matmul — that overlap is why CP works.

## Where this goes

- Topic 09 — composing CP with TP for long-context training
- Topic 10 — torchtitan CP recipe; one config switch turns it on
- Real production CP uses FlashAttention-3's CP mode (or NVIDIA Transformer Engine's CP attention) — much faster than the hand-rolled scoring here
