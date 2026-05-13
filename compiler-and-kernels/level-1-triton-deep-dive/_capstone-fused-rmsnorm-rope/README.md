# Capstone — Fused RMSNorm + RoPE vs Liger-Kernel

Every LLaMA-shaped model passes every token through an RMSNorm and then through RoPE (rotary position embedding) inside attention, and again before the MLP. Unfused, this is four separate kernels per layer per token, each one round-tripping the residual stream through HBM. Fused, it's one kernel, one HBM round-trip per occurrence. For decode-heavy workloads this is one of the highest-impact fusions you can do.

You build the fused kernel, benchmark it against Liger-Kernel's production version, and write up where you matched and where you fell short. Match Liger within ±5% of HBM peak utilization and you have produced production-grade code.

This capstone is the synthesis of everything in Level 1:
- The memory-bound elementwise+reduction template from sub-module 03 (RMSNorm bandwidth journey)
- The autotune-with-pruning discipline from sub-module 04
- Tensor descriptors from sub-module 04 (if on Hopper+)
- The persistent grid pattern from sub-module 06 (optional but recommended for the final version)

## What RoPE does, briefly

RoPE applies a position-dependent rotation to pairs of features in the input. For position `p` and feature index `i`, the rotation angle is `theta_i = p / base^(2i/d)` (typically `base = 10000` or `500000` for long context). Two consecutive features `(x[2i], x[2i+1])` are rotated to:

```
y[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
y[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
```

Elementwise per token, no inter-token dependency. The `cos` and `sin` tables can be precomputed once. The total work is roughly `2 * H` multiplies + `2 * H` adds per token — trivial compute, all the cost is reading and writing the tensor.

## What you fuse

The fused kernel reads `x` once, computes RMSNorm on it (one pass, online stats), then applies the RoPE rotation using the precomputed `cos` and `sin` tables, then writes the output. All in registers, all in one kernel, one HBM round-trip for `x` and the output. The norm weight and the cos/sin tables are loaded once per program and reused if you make the kernel persistent.

That's it. The kernel is small (~60 lines). The interesting part is autotuning it correctly and proving the bandwidth claim.

## What to do

1. Read [`CONCEPTS.md`](CONCEPTS.md) — the math (briefly), the fusion pattern, and what Liger-Kernel did.
2. Read Liger-Kernel's [`src/liger_kernel/ops/rms_norm.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py) and [`src/liger_kernel/ops/rope.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py). These are short. Make notes in `notes.md` of three things you'd lift.
3. Write your own fused kernel: [`fused_rmsnorm_rope.py`](fused_rmsnorm_rope.py). Forward-only is the minimum bar. The backward pass is a stretch goal — start with forward, get the numbers, then come back.
4. Run [`correctness.py`](correctness.py) — verify against a pure-PyTorch reference across multiple shapes including non-power-of-2 hidden dims and a long sequence (`S=8192`).
5. Run [`benchmark.py`](benchmark.py) — produce the comparison table: eager, `torch.compile`, Liger-Kernel, and three of your variants (no autotune, autotuned, persistent + autotuned).
6. Run [`profile.py`](profile.py) — confirm the HBM byte count using `triton.profiler.proton`. The trace should show: one read of x (per program), one read of w (once per persistent program, amortized), one read of cos/sin (once, amortized), one write of output. If you see more, you have a bug.
7. Write [`report.md`](report.md). Three sections: numbers, comparison to Liger, what you'd change next. The report is the deliverable for this capstone — code that runs is the table stakes; the report is what proves you understood it.

## Hardware

Free Colab T4 is enough to do the full capstone. The relative gap between unfused and fused is roughly the same across hardware (3-5×); the absolute % of peak HBM bandwidth varies. On H100 you can also turn on warp specialization for an additional 5-10% — instructions in `CONCEPTS.md`. Not needed to meet the bar.

## The bar

| Tier | Standard |
|---|---|
| Acceptable | Fused kernel correct, beats unfused by 3×+, within 25% of Liger on % HBM peak |
| Good | Within 10% of Liger on % HBM peak |
| Production-grade | Within 5% of Liger on % HBM peak, with report explaining your numbers |
| You found something | Beat Liger and triple-checked the measurement |

If you "beat Liger" with default settings, you measured wrong. Check: same dtype, same shape, same eps, both warmed up properly. After triple-checking, if you actually did beat them, write the report carefully — they will sometimes accept patches.

## Files in this folder

- `CONCEPTS.md` — the math + fusion pattern + key Liger-Kernel decisions
- `fused_rmsnorm_rope.py` — your kernel (you write the fused part; scaffolding is provided)
- `reference.py` — pure-PyTorch reference RMSNorm + RoPE (the correctness baseline)
- `correctness.py` — runs your kernel and the reference across multiple shapes
- `benchmark.py` — full comparison table runner
- `profile.py` — proton-based HBM-traffic verification
- `report.md` — your writeup template (this is the deliverable)

## After you ship this

You will have written and benchmarked a real LLM inference kernel against a production reference. The exact same template — read input once, do the fused math in registers, autotune correctly, optionally persistent — applies to: fused SwiGLU MLP, fused residual+norm, fused QKV projection + RoPE (the next-level fusion that vLLM and SGLang use), fused softmax-with-bias for attention.

Level 2 of this track takes the same kernel you wrote here and shows you how to drop it into a `torch.compile`-managed model graph. Level 3 takes the underlying online-softmax idea from sub-module 02 and rebuilds FlashAttention from it.

You're ready.
