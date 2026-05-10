# 07 — Optimization Case Study

## What this topic is

End-to-end exercise: take a deliberately slow PyTorch model, profile it, hypothesize the bottleneck, apply ONE fix, measure the delta. Repeat.

This is the muscle memory you've been building all week. It's also the entire job description for a GPU performance engineer in 2026: profile, name the regime, predict the win, fix, measure, write up.

## The canonical workflow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. BASELINE                                                  │
│    Pick a deterministic workload. Lock seeds. Record:        │
│      - latency / throughput                                  │
│      - trace (torch.profiler)                                │
│      - top kernel SOL (ncu)                                  │
│      - peak memory                                           │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. PROFILE                                                   │
│    torch.profiler first (cheap, fast).                       │
│    nsys for timeline gaps.                                   │
│    ncu for the one kernel that dominates.                    │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. HYPOTHESIZE                                               │
│    Name the regime: compute / memory / overhead / comm.      │
│    Name the specific cause. Write it down.                   │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. PREDICT                                                   │
│    "If I fuse these three ops, I save N HBM round-trips,     │
│     ≈ X% of step time."                                      │
│    The number matters. Predicting before measuring forces    │
│    you to build a model of the system.                       │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. FIX                                                       │
│    Minimal change. ONE variable.                             │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. MEASURE                                                   │
│    Same workload, same methodology.                          │
│    Compare directly. Confirm the kernel-level metric         │
│    actually moved (e.g., HBM bytes really dropped).          │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. WRITE UP                                                  │
│    PR / report description: baseline, hypothesis, fix,       │
│    before/after, trace screenshot.                           │
└──────────────────────────────────────────────────────────────┘
                          ↓
                    repeat for next bottleneck
```

What separates real optimization from cargo-culting: **step 4**. Predict the win before measuring. If your prediction is consistently right, you understand the system. If it's wrong, your mental model is broken.

## The deliberately-slow model — what to use

Take a small fine-tuning script with several known anti-patterns:

1. **Eager attention** (no FlashAttention)
2. **Unfused RMSNorm** (PyTorch's `nn.LayerNorm` instead of fused)
3. **Synchronous H2D copies** in the dataloader
4. **No `torch.compile`**
5. **Unfused AdamW**
6. **`.cpu()` calls inside the training loop** (force-syncs the GPU)
7. **`batch_size=1` dataloader with no `pin_memory`**

Each of these has a known cost and a known fix. Apply fixes one at a time and measure each delta.

## Worked example — what your case study should look like

### Baseline (anti-patterns enabled)

```
Workload: Qwen2.5-0.5B fine-tune, batch 4, seq 256, A100 GPU
Per step: 142 ms
Throughput: 7,200 tokens/sec
Top kernel: aten::native_layer_norm (12% of step time)
GPU SOL on top kernel: 18% memory, 4% compute → memory-bound
GPU utilization (nvidia-smi): 47%
```

### Hypothesis 1: dataloader-bound

Profile shows GPU idle 30-40% of step. CPU is busy in DataLoader. Diagnosis: classic dataloader bottleneck.

**Predicted impact**: with `num_workers=4` and `pin_memory=True`, dataloader stops being a bottleneck. Should drop step time by ~30%.

### Fix 1

Change `DataLoader(num_workers=0)` → `num_workers=4, pin_memory=True, persistent_workers=True`.

### Measure 1

```
Per step: 100 ms (was 142)        → 30% faster ✓ matches prediction
GPU SOL: same kernels, same %, just less idle time.
```

### Hypothesis 2: attention is eager

`torch.profiler` shows the attention block taking 35% of compute. Looking at the kernels: many small `bmm`/`softmax` calls instead of one fused FA. Diagnosis: PyTorch is using eager attention.

**Predicted impact**: `F.scaled_dot_product_attention` dispatches to FA2 on Ampere/Hopper. Should reduce attention time by 2-3×, total step time by ~15%.

### Fix 2

Replace eager attention with `F.scaled_dot_product_attention(q, k, v)`.

### Measure 2

```
Per step: 87 ms (was 100)         → 13% faster ≈ prediction
Top kernel changed: now FA2's flash_fwd dominates
GPU SOL on FA2: 65% compute, 30% memory → balanced
```

### Hypothesis 3: optim is unfused

The `optim_step` block in trace is 12 ms (14% of step). Long sequence of tiny per-tensor kernels.

**Predicted impact**: fused AdamW collapses ~100 small kernels into one. Should drop optim_step to ~3 ms.

### Fix 3

`AdamW(fused=True)`.

### Measure 3

```
Per step: 78 ms (was 87)          → 10% faster ✓
optim_step: 3 ms (was 12)         → 4× faster as predicted
```

### Hypothesis 4: compile the model

Trace shows the model has many small pointwise ops (gelu, residual adds, etc.) that aren't fused.

**Predicted impact**: torch.compile fuses elementwise sequences. Should reduce step time by another 10-20%.

### Fix 4

`model = torch.compile(model)`.

### Measure 4

```
Per step: 65 ms (was 78)          → 17% faster ✓
Kernel count: 80 → 35 (fused into bigger Triton kernels)
```

### Final results table

| Step | Fix | ms/step | speedup | tokens/sec |
|---|---|---|---|---|
| 0 | Baseline | 142 | 1.00× | 7,200 |
| 1 | num_workers + pin_memory | 100 | 1.42× | 10,240 |
| 2 | F.scaled_dot_product_attention | 87 | 1.63× | 11,770 |
| 3 | fused AdamW | 78 | 1.82× | 13,128 |
| 4 | torch.compile | 65 | 2.18× | 15,754 |

**Overall**: 2.18× faster, no model change, no quality drop.

## The writeup

The artifact this topic produces is `reports/case-study.md`. Use the template:

```markdown
# Case Study: Optimizing <model> on <hardware>

## Setup
- Model: ...
- Workload: ...
- Hardware: ...
- Methodology: warmup, lock seed, X steps measured, ...

## Baseline
- ms/step: ...
- tokens/sec: ...
- Top kernel: ... (X% of step)
- GPU SOL: ...

## Findings

### Finding 1: <name>
**Hypothesis**: ...
**Predicted impact**: ...
**Fix**: ...
**Measured impact**: ... (matches prediction? differs because...?)

### Finding 2: ...

[repeat]

## Final results
| Step | Fix | ms/step | speedup |
| ... | ... | ... | ... |

## What I'd try next
[remaining bottlenecks not yet addressed]
```

This is the systems-paper format from the curriculum's outer README. Same shape as Tri Dao's FA writeups, vLLM PR descriptions, Together AI optimization blogs.

## Why this format wins

It forces you to:

1. **Establish a baseline** — anchors all comparisons
2. **State predictions** — falsifiable; either you're right or your model is wrong
3. **One variable at a time** — each delta is interpretable
4. **Match measured to predicted** — calibrates your intuition

After 5-10 case studies, your predictions get consistently accurate. *That's* the skill.

## The 2026 examples to read

- vLLM PR descriptions for the V1 scheduler, chunked prefill tuning, FP8 KV cache
- SGLang zero-overhead scheduler PR
- FlashInfer's RoPE-fused kernel writeup (28-30% latency reduction with roofline justification)
- Red Hat: 5 steps to triage vLLM performance — https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance
- Henry Ko, Navigating Nsight Systems — https://henryhmko.github.io/posts/profiling/profiling.html
- Modal blog — https://modal.com/blog/

Each one is a worked case study. Read them and notice the pattern: baseline → hypothesis → fix → measured delta → writeup.

## Pitfalls

1. **Skipping the prediction step.** Optimization without prediction is gambling. You'll occasionally win, but you won't learn.
2. **Multiple fixes at once.** Can't tell which one worked. Always one at a time.
3. **Different methodologies before/after.** The baseline must be reproducible. Same warmup, same input, same measurement code.
4. **Cherry-picking favorable runs.** Run 5+ trials, report median.
5. **Stopping at the first improvement.** Real systems have layered bottlenecks. After fix 1, profile again — fix 2 is now visible.
6. **Optimizing kernels that aren't on the critical path.** A kernel that's 5% of total time can be 100% sped up and you'd save 5%. Always profile to find the dominant kernel first.

## What you should be able to do after this topic

- Take any slow PyTorch workload and produce a credible optimization plan from a profile
- Predict the impact of a fix before implementing it (within 30% accuracy)
- Apply Topics 01-06 fluently (the right tool for the right question)
- Write a perf case-study report that someone else can follow

## References

- Horace He — Making Deep Learning Go Brrrr — https://horace.io/brrr_intro.html (the foundation)
- Henry Ko — Navigating Nsight Systems — https://henryhmko.github.io/posts/profiling/profiling.html
- Red Hat: 5 steps to triage vLLM performance — https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance
- Stas Bekman — ML engineering performance — https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md
- Modal GPU Glossary — https://modal.com/gpu-glossary/
- TorchTitan paper — https://arxiv.org/html/2410.06511v1
