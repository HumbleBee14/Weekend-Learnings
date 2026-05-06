# 04 — Triton Intro

## Files

- `CONCEPTS.md` — Triton vs CUDA C++, autotune, what changed in 3.2/3.3, where production teams use it
- **`WHY-PYTHON-FOR-KERNELS.md`** — read this first if you're surprised this level isn't all C++. Industry context, language tradeoffs, and what each company actually uses for new kernel work.
- `vector_add.py` — the hello world. ~30 lines of Python, equivalent to topic-02's vector_add.cu.
- `matmul.py` — full matmul with autotune, group-M scheduling, fp16 → tensor cores. ~80 lines, hits ~95% of cuBLAS.
- `fused_softmax_times_y.py` — small fusion example: `softmax(x) * y` in one kernel. Shows the HBM-round-trip-elimination story.

## Quickstart

```bash
pip install triton torch
python vector_add.py
python fused_softmax_times_y.py
python matmul.py            # first run is slow (autotune); subsequent runs fast
```

## What you should see

`vector_add`: ~250 GB/s on T4, ~1+ TB/s on A100, ~3 TB/s on H100. Bandwidth-bound, fully coalesced.

`fused_softmax_times_y`: triton ≈ 2× the bandwidth of unfused PyTorch, because the fused kernel eliminates one intermediate HBM write+read.

`matmul`: on A100 fp16, expect ~250 TFLOPS triton vs 290 TFLOPS cuBLAS = ~85–95%. Triton picks WGMMA on H100 automatically — same source, different output. On T4 you'll see lower numbers because T4 has older tensor cores.

## Try

- **Compare `matmul.py` to your CUDA C++ matmul from Topic 3.** Triton is shorter and (on Ampere FP16) faster — because the autotuner picks better configs than your hand-coded ones.
- **Print the autotune choice.** Add `print(matmul_kernel.best_config)` after a call to see what was picked.
- **Run on AMD ROCm if you have access** (RunPod MI300X). Same `.py` file. The autotuner picks different optimal configs.
- **Fuse a third op.** Write `fused_softmax_y_relu(x, y)` that adds `relu()` after the multiply — still one kernel, one HBM round trip.

## Tutorials worth reading after this

The official Triton tutorials at https://triton-lang.org/main/getting-started/tutorials/ go further:
- `02-fused-softmax.html` — softmax alone, with a pretty discussion of when fusion helps
- `06-fused-attention.html` — minimum FlashAttention in Triton (the bridge to Topic 6)
- `09-persistent-matmul.html` — persistent kernel pattern + autotune at scale

For 2026 advanced patterns: Liger-Kernel ([github](https://github.com/linkedin/Liger-Kernel)) and the vLLM Triton MoE kernels are the production references.

## Where this goes

Topic 5 is the memory hierarchy in depth — it explains *why* fusion wins (the 6× SMEM-vs-HBM bandwidth gap). Topic 6 applies all of this to FlashAttention.
