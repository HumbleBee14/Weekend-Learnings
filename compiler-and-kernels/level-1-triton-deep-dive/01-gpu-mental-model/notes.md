# notes.md — your work

## Your GPU's specs

Look up the spec sheet for whatever GPU you'll be running on in the rest of this level (Colab T4 if free, A100 / H100 / B200 if cloud, MI300X if AMD). Fill in:

- GPU model: 
- Number of SMs (or CUs): 
- SRAM per SM (KB): 
- HBM total (GB): 
- HBM bandwidth (GB/s): 
- FP16 / BF16 tensor-core TFLOPS: 
- FP32 (non-tensor-core) TFLOPS: 
- Ridgeline (tensor-core TFLOPS / HBM GB/s) in FLOP/byte: 

The ridgeline number is the one you'll reach for repeatedly — it's the dividing line between memory-bound and compute-bound on your hardware.

## Diagnostic answers (closed book, in your own words)

### Q1. Smallest unit of execution and why it matters



### Q2. Grid of 8 on a 132-SM H100



### Q3. RMSNorm arithmetic intensity, memory-bound or compute-bound



### Q4. Warp divergence cost, example, mitigation



### Q5. TMA and the pattern it enables



### Q6. What `tl.dot` lowers to and why it matters



### Q7. Three reasons a memory-bound kernel hits 15% of peak



### Q8. CPU L1 vs GPU SRAM analogy — where it holds, where it breaks



## After comparing to the answer key

How many did you get substantially right (closed-book): __ / 8

Topics to re-read:


Things you understood for the first time:
