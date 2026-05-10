# 10 — torchtitan or Megatron

## Files

- `CONCEPTS.md` — pick one (torchtitan recommended), what to profile, reference links
- `mini_titan_config.toml` — small torchtitan config: FSDP2, debugmodel size, async DCP, profiler on, runs on 2 GPUs

## Quickstart

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan && pip install -e .
# Download a tokenizer — see torchtitan README. The repo also ships c4_test for smoke runs.
torchrun --standalone --nproc_per_node=2 \
    -m torchtitan.train --job.config_file ../mini_titan_config.toml
```

## Expected output

```
[INFO] Model: llama3 debugmodel (d_model=288, n_layers=6, n_heads=8) ~6M params
[INFO] FSDP2 wrapping with shard mesh of size 2
[INFO] step:  10  loss: 8.234  tps: 12,400  mfu: 18.2%
[INFO] step:  50  loss: 6.812  tps: 18,900  mfu: 27.8%
[INFO] step: 100  loss: 5.940  tps: 19,200  mfu: 28.1%   [checkpoint async-saved]
[INFO] step: 200  loss: 5.371  tps: 19,300  mfu: 28.2%
```

## Try

- Switch `tensor_parallel_degree = 2` if you have 4+ GPUs. Compare tok/s and memory.
- Enable `enable_async_tensor_parallel = true`. On compute-bound configs you'll see ~10% throughput uplift.
- Enable `enable_float8_linear = true` on Hopper. Watch MFU rise.
- Switch `pipeline_parallel_schedule = "ZBVZeroBubble"` (if your torchtitan version has it). Compare bubble fraction.

## Build steps for the rest of Level 6

The trained checkpoint (`./outputs/mini_titan/checkpoint/`) is the artifact. Levels 12 and 13 reuse this run as the harness for failure injection and async-checkpoint timing.

## Where this goes

- Topic 11 — straggler injection on this same training loop
- Topic 12 — kill a rank during this training loop; recover via Comm Shrink + the async checkpoint
- Topic 13 — measure async DCP save time on this loop
- Level 7 — `mini-platform` loads this checkpoint and serves it
