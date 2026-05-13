# notes.md — first-kernel observations

## Setup

GPU you ran on:
Triton version (`triton.__version__`):
PyTorch version (`torch.__version__`):

## 01_vector_add.py results

Throughput observed: ___ GB/s (triton) vs ___ GB/s (torch eager)

For your GPU, peak HBM bandwidth is roughly ___ GB/s. Your kernel reached ___% of peak.

If you're well below peak (<60%), the most likely cause is `BLOCK_SIZE` too small or too large. We re-visit this in sub-module 03.

## 02_scale_add.py results

Speedup of fused Triton vs eager PyTorch: ___x

What changed in the kernel vs vector add? (one sentence)

## 03_softmax_row.py results

Max diff vs `torch.softmax`: ___
Throughput on (2048, 4096): ___ GB/s

## 04_softmax_compare.py — the table

Paste the output table here:

```

```

## Observations

Where does eager torch beat your Triton kernel?

Where does `torch.compile` beat your Triton kernel?

Where does your Triton kernel hold its own?

## Takeaways before moving to sub-module 03

In a single sentence: what does kernel fusion actually save you?


Where would you intuitively expect autotune to help your kernel most? (Sub-module 03 will confirm.)


Anything that surprised you:
