# 04 — Tiled matmul and autotune that doesn't waste your money

GEMM (general matrix multiply) is the shape every learner has to understand because every dense transformer operation — attention's QK^T and PV products, the MLP up/down projections, the embedding layer, the output head — is a GEMM variant. The runtime of an LLM is dominated by GEMMs (compute-bound when batch is large, memory-bound for the weight matrices when batch is small). Getting matmul right is most of getting kernels right.

You write three matmul kernels in increasing sophistication, then turn on proper autotuning. By the end, your kernel should be within ~10% of `torch.matmul` (which dispatches to cuBLAS) on your hardware. That's the bar for "you understand GEMM."

Time budget: 3–4 hours. Free Colab T4 works for everything; H100 makes the TMA version much faster but isn't required.

## The three kernels

| File | Technique | Expected vs cuBLAS |
|---|---|---|
| `01_tiled_matmul.py` | Standard tiled GEMM with `tl.load` + `tl.dot` | 25–40% |
| `02_tma_matmul.py` | Same tiling, but using `tl.make_tensor_descriptor` | 50–80% on Hopper+; ~same on pre-Hopper |
| `03_autotuned.py` | Adds `@triton.autotune` + `early_config_prune` | 75–95% |

The autotune file is the keeper — most production matmul kernels (vLLM, SGLang, Liger) look roughly like `03_autotuned.py` with minor variations.

## What to do

1. Read [`CONCEPTS.md`](CONCEPTS.md). Make sure you understand the tile loop, the K-reduction, what `tl.dot` does, and what `tl.make_tensor_descriptor` is for.
2. Run [`01_tiled_matmul.py`](01_tiled_matmul.py). It should hit ~30% of `torch.matmul`. The number is unimpressive on purpose.
3. Run [`02_tma_matmul.py`](02_tma_matmul.py). On Hopper or Blackwell, expect 50–80%. On A100 / T4 / RTX 4090 (pre-Hopper), expect roughly the same as 01 — TMA falls back to regular loads.
4. Run [`03_autotuned.py`](03_autotuned.py). This is the one that should approach cuBLAS. The first call takes minutes (autotuning); subsequent calls are fast.
5. Run [`compare_with_inductor.py`](compare_with_inductor.py) to see how `torch.compile`'s Inductor-generated Triton compares to yours on the same shape. Inductor will be slightly faster or slightly slower; either outcome teaches you something.
6. Read your `03_autotuned.py` winning config in your `notes.md`. Write three sentences explaining *why* that specific config won — not which it was, but why. (Hint: tile sizes that fit the tensor-core fragment, register usage that doesn't spill, K-blocking that balances reuse vs. occupancy.)

## What this sub-module does NOT cover

- Split-K (parallel reduction across K dimension) — relevant for tall-skinny matmul, deferred to Level 5 (kernel fusion patterns) where it earns its own treatment.
- The Hopper WGMMA / Blackwell tcgen05 tensor-core instructions directly — those are accessed via `tl.dot` and we let the compiler handle them. The CuTe-DSL track (Level 4) goes one level lower.
- Batched matmul, grouped GEMM — these reuse the same techniques; we don't redo them. The PyTorch persistent grouped-GEMM blog post is in the resources and worth reading after you finish this sub-module.

## Files in this folder

- `CONCEPTS.md` — tile loop, tensor cores, `tl.dot`, autotune mental model
- `01_tiled_matmul.py` — basic tiled matmul
- `02_tma_matmul.py` — TMA-descriptor-based matmul
- `03_autotuned.py` — autotuned, pruned, the keeper
- `compare_with_inductor.py` — your kernel vs `torch.compile`-generated kernel
- `notes.md` — observations template

## Where this goes next

Sub-module 05 takes the TMA matmul from 02 and turns on warp specialization with one line — the producer/consumer split that made FlashAttention-3 fast. The matmul becomes ~1.3–1.5× faster on Hopper. The pattern you learn here generalizes to every fast modern kernel.

Sub-module 06 takes the autotuned matmul from 03 and makes it persistent — fixed grid, CUDA-graph compatible. That's the form vLLM ships.

The capstone fuses the RMSNorm template from sub-module 03 with a small per-element rotation (RoPE) and benchmarks against Liger-Kernel. The matmul work in this sub-module isn't directly used in the capstone, but the autotune-with-pruning discipline is.
