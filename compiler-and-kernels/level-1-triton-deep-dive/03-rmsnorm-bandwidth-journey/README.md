# 03 — The RMSNorm bandwidth journey

This is the heart of Level 1. You write the same operator — RMSNorm — five times, each version applying one new technique. You watch a single number, *fraction of peak HBM bandwidth*, climb from roughly 11% to 88%+ on your hardware. Each step teaches a generalizable lesson about why fast memory-bound kernels look the way they do.

By the end you have the canonical template for every elementwise-with-reduction kernel: LayerNorm, RMSNorm, GeGLU/SwiGLU, residual fusions, the entire class of "read a row, do a reduction, write a row" operations that show up dozens of times per transformer block.

Time budget: 3–4 hours of focused work. Free Colab T4 is sufficient.

## RMSNorm in 30 seconds

RMSNorm — root-mean-square normalization, used in LLaMA, Mistral, Qwen, GPT-NeoX — computes per row:

```
rms(x) = sqrt(mean(x_i^2) + eps)
y_i    = (x_i / rms(x)) * weight_i
```

Each row of length `H` (the hidden dimension, e.g. 4096) is normalized independently. The operation reads `H` input elements, the per-row weight vector of `H` elements, and writes `H` output elements. It does about `2H` FLOPs (one square per element, one reduction step, one division per element). Arithmetic intensity is around 0.5 FLOP/byte in fp16 — deeply memory-bound on every GPU made in the last decade. The only thing that matters is HBM bandwidth utilization.

This makes RMSNorm the perfect operator to learn the bandwidth game on. Every step in the journey moves the needle on one specific cause of bandwidth loss.

## The five steps

| File | What changes from the previous version | Expected % HBM peak |
|---|---|---|
| `01_naive.py` | First attempt: one program per row, naive load/store | 10–15% |
| `02_vectorized.py` | Vectorized loads (wider tiles, fewer transactions) | 25–35% |
| `03_single_pass.py` | Compute the reduction and normalization in one pass | 50–60% |
| `04_autotuned.py` | `@triton.autotune` with `early_config_prune` | 70–80% |
| `05_persistent.py` | Persistent kernel: each SM handles multiple rows | 80–90% |

The same data drawn as a staircase — what each step buys:

```
   % of peak HBM bandwidth

   100% ┤
    90% ┤                                                    ┌──────────┐
    80% ┤                                       ┌────────────┤    05    │  ← persistent
    70% ┤                                       │     04     │  weights reused
    60% ┤                          ┌────────────┤  autotune  │  via L2
    50% ┤                          │     03     │  + prune   │
    40% ┤             ┌────────────┤ single-pass│            │
    30% ┤             │     02     │ HBM traffic│            │
    20% ┤   ┌─────────┤ vectorized │   halved   │            │
    10% ┤   │   01    │ wider tile │            │            │
     0% ┤   │  naive  │  + cache   │            │            │
        └───┴─────────┴────────────┴────────────┴────────────┴──────────►
            tile-too- │  saturate  │   one      │  pick the  │ amortize
            small,    │  the bus   │   read     │  right     │ weight +
            2 passes  │            │            │  config    │ schedule
```
*Each step closes one specific cause of bandwidth loss. The lesson is which technique each gap demands — not the absolute numbers.*

The numbers above are rough targets — your hardware will vary. The *direction* and the *ratio between steps* should be similar. If you don't see meaningful improvement at one of the steps, something is off in your measurement (most often: not warming up, or measuring before correctness is verified).

## What to do

1. Read [`CONCEPTS.md`](CONCEPTS.md) — the lessons each step teaches, before you run them.
2. Run [`01_naive.py`](01_naive.py). Record the number in `notes.md`.
3. Run [`02_vectorized.py`](02_vectorized.py). Look at the diff vs 01. The lesson is in the diff — write it down in your own words.
4. Repeat for 03, 04, 05.
5. Run [`benchmark_all.py`](benchmark_all.py) at the end to produce one table comparing all five versions plus eager PyTorch, `torch.compile`, and Liger-Kernel's RMSNorm.
6. Profile your fastest version with `triton.proton` (instructions in `CONCEPTS.md`). Confirm: one DRAM read of the input, one DRAM read of the weight, one DRAM write of the output. If you see more, you have a bug.

The discipline: **don't move to the next step until the previous one runs correctly on your hardware and you understand why it scored where it did.** The journey is the point; rushing to step 5 with broken understanding of step 2 wastes the whole exercise.

## Reading Liger-Kernel's version after you're done

Once you've done your five steps, open Liger-Kernel's [`src/liger_kernel/ops/rms_norm.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py). Read it. You should now recognize every pattern: where they autotune, how they handle backward pass (your version is forward-only), their `early_config_prune`-equivalent. Write three things in your `notes.md`:

1. Something they do that you didn't.
2. Something you do similarly to them.
3. Something that surprised you about their implementation.

If you can fill in those three things with substance, you have read a production kernel critically. That skill is what this level is training.

## Files in this folder

- `CONCEPTS.md` — the lesson behind each of the five steps, in order
- `01_naive.py` — first attempt
- `02_vectorized.py` — wider loads, fewer HBM transactions
- `03_single_pass.py` — fuse the reduction with the elementwise step
- `04_autotuned.py` — `@triton.autotune` + `early_config_prune`
- `05_persistent.py` — persistent kernel pattern
- `benchmark_all.py` — runs all five plus references, prints comparison table
- `profile_with_proton.py` — shows how to use `triton.proton` to verify HBM traffic
- `notes.md` — your observations template

## Where this goes next

Sub-module 04 takes the same ideas — autotune with pruning, the right tile shapes for the hardware, sometimes persistence — and applies them to GEMM. The vocabulary you learn here (HBM bandwidth, tile size, fusion, pass count) becomes the vocabulary for every kernel discussion you'll have for the rest of this track.
