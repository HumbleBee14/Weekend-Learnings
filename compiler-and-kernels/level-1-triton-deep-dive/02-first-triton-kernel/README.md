# 02 — Your first Triton kernel

Three kernels, in order. Each one introduces the next piece of the Triton language and tests your mental model from sub-module 01. By the end, you can write `@triton.jit` functions and you know the shape of every common pattern.

Total time: about 90 minutes. Hardware: any CUDA GPU. Free Colab T4 is plenty.

## What to do

1. Set up your environment (see `setup.md`). Verify Triton + PyTorch work with the smoke test at the bottom of that file.
2. Run [`01_vector_add.py`](01_vector_add.py). Read every line of the kernel. The docstring walks through what each construct does.
3. Run [`02_scale_add.py`](02_scale_add.py). Compare its kernel to vector add — what changed? Write your answer in `notes.md`.
4. Run [`03_softmax_row.py`](03_softmax_row.py). This one introduces online softmax — the same idea that powers FlashAttention. Read the derivation in `CONCEPTS.md` before reading the kernel.
5. Bonus: run [`04_softmax_compare.py`](04_softmax_compare.py) to compare your Triton softmax to `torch.softmax` and `torch.compile`. The numbers will surprise you the first time.

Each `.py` file ships a correctness check (compare to PyTorch eager) and a `triton.testing.do_bench` measurement. You should see correct outputs and reasonable timings — what counts as "reasonable" depends on your hardware; expected numbers for T4, A100, H100 are in `notes.md`.

## What each file teaches

| File | New language constructs | Concept |
|---|---|---|
| `01_vector_add.py` | `@triton.jit`, `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`, `mask=` | Mapping Triton programs to tiles |
| `02_scale_add.py` | Multiple inputs, scalar args, inline arithmetic | The "elementwise" pattern |
| `03_softmax_row.py` | `tl.max`, `tl.sum`, `tl.exp`, row-per-program | Online softmax + reductions |
| `04_softmax_compare.py` | (no new kernel) | Benchmarking discipline |

## Files in this folder

- `setup.md` — install Triton + PyTorch + verify GPU access
- `CONCEPTS.md` — what tiles are, what masks mean, online softmax derivation
- `01_vector_add.py` — runnable kernel + correctness check + benchmark
- `02_scale_add.py` — runnable
- `03_softmax_row.py` — runnable, row-major softmax with online stats
- `04_softmax_compare.py` — benchmarks 03 against eager and `torch.compile`
- `notes.md` — your observations template

## Where this goes next

Sub-module 03 takes the row-softmax pattern and applies it to RMSNorm, then evolves it through five versions watching one number (% of peak HBM bandwidth) climb from 11% to 88%. The vocabulary you learn here is everything you'll need there.
