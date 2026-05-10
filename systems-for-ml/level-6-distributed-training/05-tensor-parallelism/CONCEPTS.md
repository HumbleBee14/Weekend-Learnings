# 05 — Tensor Parallelism

TP splits one matmul across GPUs. Megatron-style. The bandwidth-heaviest of the parallelism axes — must stay inside the NVLink domain.

## What TP does, mechanically

A transformer block has two big matmul groups:

```
attention: Y = (X · W_qkv) → softmax(QK^T)V → · W_o
mlp:       Y = SiLU(X · W_1) · W_2
```

TP splits each weight matrix across `tp` ranks. Two patterns:

### Column-parallel + Row-parallel pair (Megatron)

```
W_1 split column-wise:  W_1 = [W_1^(0) | W_1^(1)]   each rank owns half columns
W_2 split row-wise:     W_2 = [W_2^(0); W_2^(1)]    each rank owns half rows

Forward MLP on rank r:
  Y_partial = SiLU(X · W_1^(r)) · W_2^(r)
  Y         = all_reduce(Y_partial)                 ← the TP collective
```

The clever part: SiLU is element-wise, so the column split passes through SiLU without communication. The all-reduce happens only at the *output* of the MLP. One collective per MLP, one per attention block.

### Why it must stay intra-NVLink

Per-block all-reduce. Per-microbatch. For a 32-layer model running 2K-token microbatches at 100 step/s, that is `32 layers × 2 collectives × 100 steps = 6400 all-reduces/sec`. Each is ~1 MB at 7B-class scale. The aggregate bandwidth demand is modest, but the *latency* matters — the MLP can't continue until the all-reduce completes.

NVLink: ~1 µs latency. IB-XDR: ~3 µs. NVLink + tree algorithm at small message size: 1.5 µs. The latency budget per all-reduce is the gating factor. Going inter-node multiplies it 3–5×.

This is why TP=8 was the historical ceiling on Hopper (one DGX has 8 GPUs in one NVLink domain). On NVL72, the ceiling rises to 72.

## Sequence parallelism — the cheap add-on

The element-wise ops in a transformer (LayerNorm, dropout, residual add) replicate work across TP ranks. That's wasted compute and memory.

Sequence parallelism splits these along the *sequence* dimension instead. The cost: an extra all-gather + reduce-scatter per block (which combine to roughly the same volume as the all-reduce — they are equivalent in total bytes). The gain: the LayerNorm and dropout activations are sharded too, freeing memory.

Megatron-LM's "sequence parallelism" patch (NeurIPS 2022) and PyTorch's `SequenceParallel` style give this. In 2026 it's table-stakes; you turn it on whenever you turn on TP.

```
without SP:                          with SP:
all_reduce after attn output         reduce_scatter after attn output
all_reduce after MLP output          (sequence-sharded activations)
                                     all_gather before next block's attn input
```

Same total bytes; activation memory drops by `tp` factor.

## Async-TP — comm-compute overlap

Standard TP: all-reduce blocks the next op. Async-TP overlaps the all-reduce (or all-gather) with the *next* matmul's compute.

```
sync TP:        ┌── matmul A ──┐ ┌── allreduce ──┐ ┌── matmul B ──┐
async TP:       ┌── matmul A ──┐
                                ┌── allreduce ──┐
                                                 └── matmul B starts on its share ──┐
                                                       (waits on remaining shares as they arrive)
```

PyTorch shipped this as `torch.distributed._functional_collectives` paired with the compiler. torchtitan and Megatron both wire it in 2026. ~10–15% speedup on TP-heavy configs.

## TP from `torch.distributed.tensor.parallel`

```python
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel
)

tp_mesh = mesh["tp"]

parallelize_module(
    block,
    tp_mesh,
    {
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(),
        "mlp.w1": ColwiseParallel(),
        "mlp.w3": ColwiseParallel(),  # SwiGLU has w1 and w3 column-parallel
        "mlp.w2": RowwiseParallel(),
        "attention_norm": SequenceParallel(),
        "ffn_norm": SequenceParallel(),
    },
)
```

The dict says "this submodule is column/row/sequence parallel on `tp_mesh`." Composes with FSDP2's `fully_shard` on a different mesh dim.

## Build steps

1. Use `tp_demo.py` to apply TP=2 to a single transformer block.
2. Print parameter shapes before and after. `attention.wq.weight` halves its column count.
3. Run forward. Confirm output shape matches single-GPU.
4. Time: TP=1 vs TP=2 on the same hardware. TP=2 wins on memory; throughput depends on whether you're matmul-bound or comms-bound. On a small model, it loses on throughput because the matmul is too small to amortize the comm.

## When TP is worth it

- Layer doesn't fit on one GPU → mandatory.
- Layer fits but is huge (50%+ of device memory) → activation pressure makes TP worth it.
- Layer fits comfortably → TP usually loses to FSDP2 alone. Don't reach for it.

## Reference

- Megatron-LM paper: [arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- Sequence parallelism (Megatron): [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198)
- Async-TP / Comm overlap: [pytorch.org/blog/training-production-ai-models](https://pytorch.org/blog/training-production-ai-models/)
- PyTorch TP API: [pytorch.org/docs/stable/distributed.tensor.parallel.html](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- torchtitan TP recipe: [github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms](https://github.com/pytorch/torchtitan/tree/main/torchtitan/parallelisms)
