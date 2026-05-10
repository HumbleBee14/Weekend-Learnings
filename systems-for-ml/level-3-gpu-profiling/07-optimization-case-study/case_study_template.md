# Case Study Template

Use this as the structure for `reports/case-study.md`. Replace placeholders with your numbers.

---

# Case Study: Optimizing TinyTransformer on <GPU>

## Setup

- **Model**: TinyTransformer (4 layers, d_model=512) — `slow_model.py`
- **Workload**: batch=4, seq_len=256, fake-data fine-tuning
- **Hardware**: <e.g. NVIDIA A100 80GB, CUDA 13.2, PyTorch 2.10>
- **Methodology**: 4 warmup steps discarded, 30 measured steps, median reported
- **Tool**: `torch.profiler` for time, `ncu --set basic -k <kernel>` for kernel SOL where useful

## Baseline (all anti-patterns enabled)

```
ms/step:        <X> ms
tokens/sec:     <Y>
final loss:     <Z>
GPU util:       <%>
Top kernel:     <name> (<X% of step>)
```

Trace screenshot: `reports/trace_baseline.png`

## Findings

### Finding 1: Dataloader-bound

**Observation**: GPU idle for ~30% of step time. CPU thread busy in `DataLoader.__getitem__`. The fake-data sleep (2ms × 4 items = 8ms per batch) blocks the GPU.

**Hypothesis**: synchronous, single-worker dataloader. Fix is num_workers + pin_memory.

**Predicted impact**: ~25% reduction in step time (current 8ms data wait becomes asynchronous).

**Fix**: `FIX_DATALOADER = True`

**Measured**: ms/step <X> → <Y>, **delta <Z>%**.

Matches prediction? Why or why not?

### Finding 2: Eager attention

**Observation**: `aten::native_layer_norm` and many small `bmm` calls dominate. Attention matmul materializes the (N×N) score matrix in HBM. Memory throughput high but compute SOL low.

**Hypothesis**: PyTorch is using eager attention. Replacing with `F.scaled_dot_product_attention` will dispatch to FA2/FA3 on this GPU, fusing the QK^T → softmax → V chain.

**Predicted impact**: attention block becomes ~2-3× faster, ~15% of step time.

**Fix**: `FIX_USE_SDPA = True`

**Measured**: ms/step <X> → <Y>, **delta <Z>%**. Top kernel changed from `bmm` to `flash_fwd_*`.

### Finding 3: Unfused AdamW

**Observation**: `optim_step` is <X>ms, dominated by long sequence of small per-tensor kernels.

**Hypothesis**: unfused AdamW. Switching to `fused=True` collapses ~100 small kernels into one.

**Predicted impact**: optim_step from <X>ms to ~<X/4>ms.

**Fix**: `FIX_FUSED_ADAMW = True`

**Measured**: ms/step <X> → <Y>, **delta <Z>%**.

### Finding 4: torch.compile not enabled

**Observation**: Kernel count is <N>; many small pointwise ops (gelu, residual adds, layernorm scales) un-fused.

**Hypothesis**: Inductor will fuse pointwise sequences. Step time should drop ~10-15%.

**Predicted impact**: <X>% reduction.

**Fix**: `FIX_COMPILE = True`

**Measured**: ms/step <X> → <Y>, **delta <Z>%**. Kernel count: <N> → <M>.

### Finding 5: Sync H2D + .cpu() in loop

**Observation**: each step has a yellow H2D bar on the critical path. Each step also has `loss.cpu()` which forces a sync.

**Hypothesis**: enable `non_blocking=True` for the H2D and defer the .item() call to outside the hot loop.

**Predicted impact**: small — these are individually <1% of step but together can be 5-8%.

**Fix**: `FIX_NON_BLOCKING_H2D = True`, `FIX_REMOVE_CPU_SYNC = True`

**Measured**: ms/step <X> → <Y>, **delta <Z>%**.

## Final results

| Step | Fix | ms/step | speedup | tokens/sec |
|---|---|---|---|---|
| 0 | Baseline | <X> | 1.00× | <Y> |
| 1 | + dataloader | <X> | <Y>× | <Z> |
| 2 | + SDPA | <X> | <Y>× | <Z> |
| 3 | + fused AdamW | <X> | <Y>× | <Z> |
| 4 | + torch.compile | <X> | <Y>× | <Z> |
| 5 | + async H2D + no .cpu() | <X> | <Y>× | <Z> |

**Final**: <X>× faster, no model change, no quality drop.

## What I'd try next

What bottlenecks remain at this point? After Topic 06's fixes, the next things to look at are usually:

- **Quantization** — drop weights to FP8 → 2× compute peak on Hopper, halved HBM traffic. Level 4 covers this.
- **Larger batch** — current step is bandwidth-limited; batch=16+ would push us closer to compute-bound. Check OOM headroom first.
- **CUDA graphs** — eliminate per-kernel launch overhead in the eager loop. Worth measuring CPU side first.
- **Multi-GPU** — if model is too big for one GPU, FSDP. Different bottleneck regime entirely (Level 6).

## Trace artifacts

- `reports/trace_baseline.png`
- `reports/trace_after_dataloader_fix.png`
- `reports/trace_final.png`
- Raw traces in `traces/`

## Lessons

What surprised you? Where did your prediction differ most from measurement? Those are where your mental model needs work.
