# 07 — Optimization Case Study

## Files

- `CONCEPTS.md` — the canonical workflow (baseline → profile → hypothesize → predict → fix → measure → write up), worked example, the 2026 industry references to read
- `slow_model.py` — a deliberately-slow TinyTransformer with 6 toggleable anti-patterns. Toggle one fix at a time and measure the delta.
- `case_study_template.md` — the writeup structure for `reports/case-study.md`

## Quickstart

```bash
pip install torch
python slow_model.py     # baseline (all anti-patterns)
```

Then iteratively edit the `FIX_*` flags at the top of `slow_model.py`, re-run, and measure each delta.

The recommended order (each one fixes the *currently dominant* bottleneck):

```
1. FIX_DATALOADER = True
2. FIX_USE_SDPA = True
3. FIX_FUSED_ADAMW = True
4. FIX_COMPILE = True
5. FIX_NON_BLOCKING_H2D = True; FIX_REMOVE_CPU_SYNC = True
```

## What you should see

Approximate numbers on an A100 — your hardware will differ:

```
Baseline (no fixes):                ~150 ms/step,  ~6,800 tokens/sec
+ dataloader fix:                   ~110 ms/step,  ~9,300 tokens/sec
+ SDPA (FlashAttention):            ~85 ms/step,   ~12,000 tokens/sec
+ fused AdamW:                      ~75 ms/step,   ~13,600 tokens/sec
+ torch.compile:                    ~58 ms/step,   ~17,600 tokens/sec
+ async H2D + remove .cpu() sync:   ~52 ms/step,   ~19,700 tokens/sec

Total: 2.9× faster, no model change.
```

If your numbers differ a lot (especially with `torch.compile` — first run includes JIT), be sure to re-warm before measuring. The script does 4 warmup steps; for `torch.compile` you might need more.

## How to do this properly

For each fix:

1. **Profile before** with `torch.profiler` (Topic 03). Save the trace.
2. **Predict** the impact in writing — quantitatively (e.g., "~25% step reduction").
3. **Apply ONE fix** (one flag).
4. **Re-run, measure**. Save the new trace.
5. **Compare**. Did the prediction match? Why/why not?
6. **Update** the case study writeup.

Don't apply two fixes at once — you lose the per-fix delta.

## The deliverable — `reports/case-study.md`

Use `case_study_template.md` as the structure. Fill in your numbers. The template enforces:

- Setup (reproducible methodology)
- Baseline numbers
- Per-fix: hypothesis + prediction + fix + measured + analysis
- Final results table
- "What I'd try next"
- Trace artifacts

This is **the artifact for Project 1**. Combined with Topic 05's `profiling-mini-serve.md`, you have evidence that justifies every Level 4 optimization choice.

## What you should be able to do after this topic

The full week of Level 3 culminates here. Test yourself:

- Pick any slow PyTorch workload you didn't write
- Profile it
- Identify the top bottleneck
- Predict the impact of *one* fix in writing, with a number
- Apply the fix
- Measure
- Be within 30% of your prediction

If you can do this, you've reached the bar for "GPU performance engineer." Everything else is reps.

## Where this goes

You now have:
- **`profiling-mini-serve.md`** (Topic 05) — characterization of `mini-serve`'s bottlenecks
- **`case-study.md`** (this topic) — the full optimization workflow demonstrated end-to-end

Level 4 is the application: every fix you'll make to `mini-serve` to create `mini-vllm` (paged KV, continuous batching, fused kernels, quantization) is justified by these two reports. You're done with profiling. Time to optimize.
