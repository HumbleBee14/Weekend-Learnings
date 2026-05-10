# 09 — 5D Parallelism Composition

## Files

- `CONCEPTS.md` — the decision tree, worked examples for 7B / 70B / 405B / MoE / long-context, why goodput replaces MFU
- `mesh_compose.py` — concrete FSDP2 + TP composition via 2-D DeviceMesh

## Quickstart

```bash
torchrun --standalone --nproc_per_node=4 mesh_compose.py
# also runs (degenerately) on 2 GPUs
torchrun --standalone --nproc_per_node=2 mesh_compose.py
```

## Expected output (4 GPUs)

```
DeviceMesh: (2,2) on 4 ranks
  dp_shard sub-mesh: DeviceMesh('cuda', [[0, 1], [2, 3]], mesh_dim_names=('dp_shard',))
  tp sub-mesh:       DeviceMesh('cuda', [[0, 1], [2, 3]], mesh_dim_names=('tp',))
output shape (8, 64, 512)  loss 0.3201
blocks[0].w1.weight: shape=(1024, 512), type=DTensor
  placements (per mesh dim): (Shard(dim=0), Shard(dim=0))
```

`Shard(dim=0)` on both mesh dims confirms the parameter is sharded twice — once for FSDP, once for TP. Effective replication factor `dp × tp = world`.

## Try

- Build a 3-D mesh `(2, 2, 2)` named `("dp_shard", "pp", "tp")` on 8 GPUs and add a manual PP split between blocks 0–1 and 2–3 (Topic 06's pattern).
- Print `mesh.get_local_rank("tp")` from each rank — confirms which TP shard this rank holds.
- Switch the `(dp, tp)` partitioning to `(world, 1)` (FSDP only) and `(1, world)` (TP only). Compare time/step. The right choice depends on model size and interconnect.

## Where this goes

- Topic 10 — torchtitan does all of this declaratively from a `.toml`
- Topic 12 — failure on a 2-D mesh: which axis can survive a rank loss? (Answer: dp_shard via Comm Shrink; tp cannot)
