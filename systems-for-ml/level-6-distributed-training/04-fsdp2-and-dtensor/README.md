# 04 — FSDP2 and DTensor

## Files

- `CONCEPTS.md` — what FSDP shards, why FSDP2 replaced FSDP1, DeviceMesh, HSDP, mixed precision, common foot-guns
- `fsdp_train.py` — the Topic 02 transformer rewrapped with `fully_shard`, DeviceMesh, MixedPrecisionPolicy, sharded DCP save

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 fsdp_train.py
```

## Expected output (2×A10)

```
step    0  loss 9.108  tok/s 12,400  peak 1.32 GB
step   10  loss 7.704  tok/s 39,200  peak 1.41 GB
step   20  loss 7.014  tok/s 40,800  peak 1.42 GB
step   30  loss 6.617  tok/s 41,100  peak 1.42 GB
step   40  loss 6.341  tok/s 41,300  peak 1.42 GB
step   50  loss 6.122  tok/s 41,400  peak 1.42 GB
sharded checkpoint written: ckpt_fsdp_step50/
```

Compare peak memory to Topic 02's DDP run. At this small scale the gap is modest (~10–20% lower) — at 7B+ scale it is the difference between OOM and not.

`ls ckpt_fsdp_step50/` should show one file per rank plus a `.metadata` file. That is sharded DCP.

## Try

- Print `model.blocks[0].self_attn.in_proj_weight` after `fully_shard`. It is a `DTensor`. Print `.placements` to see `(Shard(dim=0),)`.
- Add `parallelize_module` to apply TP to one block; init mesh as `(2, 1)` with names `("dp_shard", "tp")`. Composition is a single line per axis.
- Switch `mesh_shape=(world,)` to a 2-D mesh `(2, world//2)` and use `init_device_mesh(... ,mesh_dim_names=("dp_replicate","dp_shard"))` to get HSDP. Pass both axes to `fully_shard`.
- Load the saved checkpoint with `dcp.load` on a different `world_size` (when you have access to another box) — this is the elastic-recovery property.

## Where this goes

- Topic 05 — TP composes with FSDP2 via DeviceMesh
- Topic 12 — failure injection on this same script
- Topic 13 — async DCP variants of the save call here
- Topic 10 — torchtitan does all of this with one config file
