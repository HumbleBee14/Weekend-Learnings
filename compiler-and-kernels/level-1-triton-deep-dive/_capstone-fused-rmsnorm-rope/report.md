# Capstone report — Fused RMSNorm + RoPE

Fill this in. The report is the deliverable for the capstone. Code that runs is the table stakes; the writeup is what proves you understood.

## Hardware and setup

GPU: 
HBM peak (GB/s): 
Triton version: 
PyTorch version: 
Liger-Kernel version (if installed): 

## The numbers

Paste the table from `benchmark.py`:

```

```

Paste the relevant proton metrics from `profile.py`:

```
dram__bytes_read per call:
dram__bytes_write per call:
dram__throughput pct:
sm__warps_active pct:
```

## Where you stood vs Liger-Kernel

Your best variant achieved __% of HBM peak.
Liger achieved __% of HBM peak.

Gap: ___% (positive = Liger faster, negative = you faster).

If positive — your kernel is _slower_ than Liger — describe the most likely cause based on what you observed. (Common: extra HBM transactions from the partner-element gather; suboptimal `BLOCK_SIZE` for your hardware; missing `tl.constexpr` casting-mode tuning.)


If negative — you beat Liger — what did you change that helped? Are you sure you measured the same operation? (Liger's `LigerRopeFunction` may be slightly different from yours in terms of cos/sin precision or interleave convention.)


## What you would do next to close the gap (or extend the lead)

Three concrete things:

1. 

2. 

3. 

## What you took away from Level 1

In a paragraph: what did you understand at the end of this level that you didn't at the start? Use specifics — not "I learned Triton" but "I understand that the win in fusion comes from cutting HBM round-trips, and I can show that with a profiler trace; I can read what Inductor emits and form an opinion; I know the difference between memory-bound and compute-bound and can identify which an operator is from its arithmetic intensity."


## Open questions you'd want to dig into in Level 2

What's the equivalent fusion that `torch.compile` would do automatically? When does the compiler beat hand-written code, and when doesn't it? Where's the line between "fuse it yourself" and "let Inductor do it"?
