# notes.md — RMSNorm bandwidth journey

## Setup

GPU: 
Approx HBM peak (GB/s): 
Triton version: 

## The journey, step by step

For each step, run the script, note the GB/s and % of peak, and write one sentence about what the previous-vs-current diff teaches you.

### Step 01 — naive
GB/s: ___      % of peak: ___
Lesson: 


### Step 02 — vectorized
GB/s: ___      % of peak: ___
What changed in the kernel vs 01:
Lesson: 


### Step 03 — single pass
GB/s: ___      % of peak: ___
What changed:
Lesson:


### Step 04 — autotuned with pruning
GB/s: ___      % of peak: ___
Best config the autotuner picked: 
What changed:
Lesson:


### Step 05 — persistent
GB/s: ___      % of peak: ___
Best config: 
What changed:
Lesson:


## benchmark_all.py — the comparison table

Paste the output here:

```

```

How does your step 05 compare to Liger-Kernel? (Within 5%, within 15%, beat them, lost to them.)


How does your step 05 compare to `torch.compile`? (Inductor will autotune internally.)


## proton profile

Did you confirm `dram__bytes_read` ≈ minimum?  yes / no
Did you confirm `dram__bytes_write` ≈ minimum?  yes / no
DRAM throughput % observed: 

If anything was off, what did you discover?


## After reading Liger-Kernel's rms_norm.py

Something they do that you didn't:


Something you do similarly:


Something that surprised you:


## The generalizable take-away

In your own words: what is the template for memory-bound elementwise+reduction kernels you can now apply to any operator?
