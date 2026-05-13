# The five lessons of the bandwidth journey

Each step in this sub-module isolates one specific cause of bandwidth loss. Read this whole document before running the scripts — knowing what to look for at each step is what makes the exercise worth doing.

## Step 1 — Naive: 10–15% of peak

The kernel: one program per row, load the row, compute the squared sum with `tl.sum(x * x)`, compute the RMS, normalize, store. Forward-pass only, fp32 for clarity.

Why it's slow:

- **`BLOCK_SIZE` chosen by intuition.** We pick 1024 or 2048 because they look reasonable. The compiler can't pick — `BLOCK_SIZE` is a `tl.constexpr`, which means it's a compile-time decision baked into the kernel. The optimal value depends on the hidden dim, the SM count, and the SRAM size. We aren't tuning it yet.
- **Two passes over the row.** First pass: compute `sum(x^2)`. Second pass: load `x` again, divide, multiply by weight, store. The row gets loaded *twice* from HBM. We're paying double bandwidth for no good reason.
- **Sub-optimal access pattern.** `tl.load(x_ptr + offsets)` is fine but the implicit transaction width may not match the row stride. On rows that aren't a multiple of the warp's preferred transaction size (128 bytes on H100, 256 on T4), we're issuing more transactions than necessary.

Expected number on T4: ~30 GB/s out of ~300 GB/s peak — about 10%. On H100: ~400 GB/s out of ~3400 — about 12%. The number is depressing on purpose. We have nowhere to go but up.

## Step 2 — Vectorized loads: 25–35% of peak

The kernel: same algorithm, but we load in wider chunks using a larger `BLOCK_SIZE` and explicit cache hint. The hardware can coalesce a `tl.load` of 1024 floats into a small number of fat transactions, but only if `BLOCK_SIZE` is big enough to occupy the memory bus and small enough to fit in registers.

What changed:

- **`BLOCK_SIZE` rounded up to the row size.** If the row is 4096, use `BLOCK_SIZE = 4096`. The whole row lives in one tile. No tiling within a row.
- **`num_warps` tuned by hand.** With a 4096-wide tile, `num_warps=8` gives each warp 128 elements of work. Too few warps (1 or 2) underuses the SM. Too many spills to SRAM. Eight is a starting point; we'll let autotune find the real winner in step 4.
- **Cache hints on the load.** `tl.load(..., cache_modifier=".ca")` requests an L1-cacheable load. For data we read once and never re-read in this kernel, `.cs` (streaming, bypass L1) is sometimes better — less pollution of the cache.

Still two passes over the row. We left the biggest single win on the table on purpose so we can isolate the next one.

The lesson: **tile size matters more than algorithm at this point.** The exact same compute, with a bigger tile and better warp count, is 2-3× faster.

## Step 3 — Single pass: 50–60% of peak

The kernel: one pass over the row. Load `x` once into registers, compute `sum(x*x)` while we have it, then do the elementwise normalize and the weight multiplication in the same register tile, then store.

What's nice: **we read the input from HBM exactly once and write the output once.** The weight is read once too. The HBM bytes moved is the theoretical minimum: `(H_in + H_weight) * dtype_size` for the load, `H_out * dtype_size` for the store. There is no algorithmic way to do less.

We jump from ~30% to ~55% by halving the HBM traffic. Same algorithm, same hardware, half the bandwidth used. This is what fusion buys you in concrete bytes.

The remaining gap to peak comes from elsewhere — config choice and per-SM occupancy. We've squeezed the algorithm; now we squeeze the launch parameters.

## Step 4 — Autotuned: 70–80% of peak

`@triton.autotune` lets you list multiple configs (different `BLOCK_SIZE`, `num_warps`, `num_stages`), and Triton compiles each variant and benchmarks them at the first call for each input shape. Subsequent calls with that shape use the winner.

The naive autotune list is "every combination of `BLOCK_SIZE ∈ {1024, 2048, 4096, 8192}`, `num_warps ∈ {1, 2, 4, 8, 16}`, `num_stages ∈ {2, 3, 4}`". That's 60 configs and many are nonsense — `num_warps=1` with `BLOCK_SIZE=8192` would spill enormously; `BLOCK_SIZE=1024` on a row of 4096 wastes the tile and needs four iterations.

`early_config_prune` lets you write a Python function that filters the config list before benchmarking. The function gets the configs and the named kernel args (including the runtime shape) and returns the pruned set. We prune:

- Configs where `BLOCK_SIZE < n_cols` and we'd need multiple iterations (less efficient for these shapes — keep the one-tile-per-row form pure)
- Configs where `BLOCK_SIZE > 2 * next_power_of_2(n_cols)` — wasted threads
- Configs where the SRAM footprint exceeds the SM's budget (register and shared memory pressure)
- Configs where `num_warps * 32` doesn't divide cleanly into the tile size

After pruning, ~6–10 configs remain. Autotuning takes seconds, not minutes. The winning config is shape-dependent — for `n_cols = 4096` it might be `(BLOCK=4096, warps=8, stages=3)`; for `n_cols = 2048` it might differ.

The lesson: **the right autotune doesn't mean searching everything. It means searching the candidates that aren't obviously bad.** Most autotune time should be spent on the 10% of configs that might win, not the 90% that won't.

## Step 5 — Persistent: 80–90% of peak

So far, each kernel launch creates `num_rows` programs and the hardware scheduler doles them out across SMs. For a batch of 32 sequences × 2048 tokens = 65,536 rows, that's 65,536 programs scheduled on 132 SMs (H100) — about 500 waves of scheduling. Each transition costs some launch overhead and some L2 cache eviction.

A persistent kernel launches *exactly* `num_SMs` programs (e.g., 132 on H100, 40 on T4) and each program processes multiple rows internally:

```python
@triton.jit
def kernel(...):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)  # equals num_SMs

    for row in range(pid, n_rows, num_pids):
        # ... process one row ...
```

Each SM picks up rows `pid, pid + num_pids, pid + 2*num_pids, ...`. The SM stays "warm" — registers stay allocated, the weight vector (which is reused across rows) can sit in L2 and serve every row that program processes.

Why this gets us into the 80–90% range:

- **No re-scheduling.** The hardware schedules 132 programs once and they run to completion. ~500 waves of scheduling become 1 wave.
- **Weight reuse from L2.** The weight vector is the same for every row in the batch. The first row on each SM pays HBM cost to load it; subsequent rows on the same SM hit it from L2. For large hidden dims this is a noticeable saving.
- **CUDA-graph friendly.** A fixed grid means the kernel captures into a CUDA graph cleanly, which we exploit in sub-module 06 and Level 2.

The last 10–20% gap to absolute peak is from things we deliberately didn't do at this level: warp specialization (sub-module 05) to hide load latency behind compute, and asymmetric pipelines optimized for specific batch shapes. Those would push us into the high 90s but cost in complexity. For RMSNorm — and for most elementwise+reduction kernels — 80–90% is the practical ceiling and we stop here. You'll see Liger-Kernel does about the same: simple algorithm, well-tuned, no fancy warp tricks.

## Profiling with `triton.proton`

Once you have your 04 or 05 version running, profile it:

```python
import triton.profiler as proton

with proton.scope("rmsnorm_v4"):
    out = rmsnorm_v4(x, weight)

proton.finalize()
```

`proton` produces a JSON trace you can view with `proton-viewer` or load into a notebook. The metrics you want to confirm:

- `dram__bytes_read.sum` ≈ `H_in * batch_rows * dtype_size + H_weight * dtype_size` — one read of the input, one read of the weights (amortized if persistent)
- `dram__bytes_write.sum` ≈ `H_out * batch_rows * dtype_size` — one write of the output
- `sm__warps_active.avg.pct_of_peak_sustained_active` > 70% — SMs are busy
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` > 80% — we're using the memory bus

If any of those look wrong, find out why before declaring victory. The classic "fast but wrong" pattern is: your kernel never actually executed (cache miss, grid size 0) and `do_bench` is measuring a no-op. Always sanity-check the output against eager before believing the timer.

## The generalizable template

After this sub-module, you have the pattern for every memory-bound elementwise+reduction kernel:

1. One program per output row (or chunk of rows for tiny shapes).
2. Whole row fits in one tile if possible. Use `tl.constexpr BLOCK_SIZE` set to `next_power_of_2(n_cols)`.
3. Single pass over the row: load → compute reduction stats → compute normalized output → store. No reloading.
4. Autotune across `(BLOCK_SIZE, num_warps, num_stages)` with `early_config_prune` to skip illegal configs.
5. Persistent grid (`num_programs = num_SMs`) if the batch is large enough that you'd otherwise launch hundreds of waves.

Apply this template directly to LayerNorm, GeGLU, SwiGLU, GroupNorm, batched residual+norm fusions. The capstone of this level (fused RMSNorm+RoPE) is exactly this template with an additional small per-element rotation tacked on.
