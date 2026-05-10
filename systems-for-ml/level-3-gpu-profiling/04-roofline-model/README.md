# 04 — Roofline Model

## Files

- `CONCEPTS.md` — the chart, computing AI for common LLM kernels (GEMM, attention, RMSNorm), 2026 hardware peaks, Horace He's three regimes, communication-bound, hierarchical roofline
- `plot_roofline.py` — generate a roofline plot for your GPU with four common LLM kernels marked

## Quickstart

```bash
pip install matplotlib numpy
python plot_roofline.py
# → roofline.png
```

Edit the `GPU_NAME`, `PEAK_TFLOPS`, `PEAK_HBM_BW_TBPS` at the top of the script for your GPU.

## What you'll see

```
H100 SXM: peak 989 TFLOPS, HBM 3.35 TB/s, ridge AI = 295 FLOP/byte

Kernel analysis on H100 SXM (ridge AI = 295):
  GEMM 4096³ BF16: AI=1365.3, achieved=750.0 TFLOPS, ceiling=989.0, util=76%  (compute-bound)
  GEMM decode (M=1): AI=1.0, achieved=2.0 TFLOPS, ceiling=3.4, util=60%  (memory-bound)
  FlashAttention N=4096: AI=512.0, achieved=600.0 TFLOPS, ceiling=989.0, util=61%  (compute-bound)
  RMSNorm (per token): AI=0.8, achieved=0.05 TFLOPS, ceiling=2.5, util=2%  (memory-bound)
```

The numbers tell the story:

- **GEMM 4096³**: well to the right of the ridge → compute-bound → close to peak. Good.
- **GEMM decode**: AI=1, memory-bound by a huge margin → why decode is slow.
- **FlashAttention**: AI=512, compute-bound on long contexts → why FA is fast.
- **RMSNorm**: AI=0.8, memory-bound → why it MUST be fused.

The roofline plot puts these on one chart so the regimes are obvious at a glance.

## Try

- **Change the GPU peaks** to A100 (312 TFLOPS, 1.94 TB/s). The ridge moves to ~161. Kernels with AI in 161-295 *change regime* between A100 and H100. Same code, different bottleneck.
- **Add an FP8 GEMM**: peaks double on H100 (989 → ~2000 TFLOPS for FP8 dense), bytes per element halve. The kernel's regime shifts.
- **Plot a real kernel**: run your kernel under `ncu --set full`, get achieved TFLOPS and AI from the report, plot the point. Where does it sit relative to the ceiling?
- **Plot multiple ceilings** (HBM, L2, SMEM bandwidth — see CONCEPTS.md for the hierarchy). A kernel that fits in L2 sits on a higher ceiling than one that doesn't.

## Mental shortcuts

You should be able to do these on a napkin without the script:

| Kernel | Quick AI estimate | Regime on H100 |
|---|---|---|
| GEMM (square, large M=N=K) | ~K (in FP16) | compute |
| GEMM (decode, M=1) | ~1 | memory |
| FlashAttention forward | ~N/2 | compute (long context) |
| RMSNorm, softmax | < 2 | memory |
| Element-wise (add, mul) | < 1 | memory |
| AllReduce | 0 (compute is zero) | comm-bound |

If you can recite this table, you can predict the bottleneck of most LLM kernels without running them.

## When this matters in real work

The roofline isn't an academic exercise. The decisions it informs:

- **"Should I write a fused kernel?"** — only worth it if the unfused version is memory-bound.
- **"Should I quantize?"** — quantization helps memory-bound kernels (less data); it can also flip a kernel's regime.
- **"Why is my decode so slow?"** — because it's at AI ≈ 1, not because the kernel is poorly written.
- **"Will this batch size help?"** — bigger batch → higher AI for decode (weights reused across the batch). At what batch does decode become compute-bound? The roofline answers this.

## Where this goes

Topic 05 applies all of Topics 01-04 to a real LLM serving workload. Topic 06 applies them to a training loop. Topic 07 is the full case study where you take a slow model, profile it, hypothesize, fix, measure delta — using the roofline to predict the win before you run the experiment.
