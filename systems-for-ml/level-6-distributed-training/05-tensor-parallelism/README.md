# 05 — Tensor Parallelism

## Files

- `CONCEPTS.md` — column/row split, sequence parallelism, async-TP, why it must stay intra-NVLink
- `tp_demo.py` — applies TP=2 to a SwiGLU MLP using `parallelize_module`; prints sharded shapes; times forward

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 tp_demo.py
```

## Expected output

```
Before TP:
  w1.weight: (4096, 1024)
  w2.weight: (1024, 4096)
  w3.weight: (4096, 1024)
After TP (each rank holds a shard):
  w1.weight: (2048, 1024)  (DTensor placement: (Shard(dim=0),))
  w2.weight: (1024, 2048)  (DTensor placement: (Shard(dim=1),))
output shape: (4, 64, 1024)  (matches single-GPU shape)
output mean : -0.001234  std: 0.4912
50 forwards: 7.84 ms total (0.157 ms/forward)
```

The `Shard(dim=0)` on `w1` is column-parallel; `Shard(dim=1)` on `w2` is row-parallel; the all-reduce happens inside `w2`'s `RowwiseParallel` wrapper.

## Try

- TP=1 vs TP=2: comment out `parallelize_module` and rerun. Compare forward time. On a small MLP, TP=2 is often *slower* — the matmul is too small to hide the all-reduce. TP wins on memory and on big matmuls.
- Replace the MLP with a real attention block. The pattern: `wq, wk, wv` column-parallel; `wo` row-parallel.
- Compose with FSDP2: extend the mesh to 2-D `(dp, tp)` and pass `mesh["dp"]` to `fully_shard` and `mesh["tp"]` to `parallelize_module`. This is the canonical Llama-3-style configuration.
- Force NVLink off (`NCCL_P2P_DISABLE=1`) and rerun the timing. Watch the all-reduce dominate.

## Where this goes

- Topic 09 — composing TP with the other axes
- Topic 10 — torchtitan applies all of this with one config switch
