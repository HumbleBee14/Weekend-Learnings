# Level 5 — Kernel Fusion Patterns

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: kernel fusion profiling report — top-3 bandwidth-bound ops, hand-fused Triton vs torch.compile

## Week goal

This week synthesizes everything from Levels 1–4. Fusion is where the kernel skills pay off in practice. By Friday you should be able to:

- Profile a full transformer forward pass and identify which ops are bandwidth-bound vs compute-bound
- Correctly decide when hand-written Triton fusion beats `torch.compile`'s automatic fusion — and when it doesn't
- Write fused kernels for the 3 most bandwidth-bound ops in a LLaMA-style model
- Understand the limits of fusion: register pressure, shared memory constraints, and when fusion hurts

## Where this fits

- **Comes after:** Levels 1–4. You need Triton (Level 1) for writing fused ops, `torch.compile` internals (Level 2) for understanding automatic fusion, attention (Level 3) for the attention-specific fusion story, and CuTe-DSL (Level 4) for GEMM epilogue fusion.
- **Comes before:** Level 6 (MLIR — which is the compiler substrate underneath all of this; after learning what fusion should look like, you learn how a compiler decides it automatically).

## 2026 state of kernel fusion

**The canonical LLM decoder fusion map.** For one transformer layer:

| Op | Fusion opportunity | Best approach |
|---|---|---|
| RMSNorm + RoPE | Fuse into one read/write | Hand-written Triton (Liger-Kernel pattern) |
| QKV projection | GEMM epilogue (bias) | CuTe-DSL / CUTLASS epilogue |
| Attention | Already fused (FA3/FA4) | FlashInfer / FlexAttention |
| Output projection | GEMM epilogue (bias) | CuTe-DSL / CUTLASS epilogue |
| MLP (gate × up → silu) | Fuse gate + silu | Triton fused SwiGLU (Liger-Kernel) |
| MLP down projection | GEMM epilogue (activation scale) | CUTLASS |
| RMSNorm (pre/post) | Fuse with next op where possible | Hand-written Triton |
| Residual add | Fuse into preceding kernel epilogue | `torch.compile` handles this |

**How torch.compile fuses.** Inductor's fusion pass identifies "pointwise islands" — sequences of elementwise ops with no intervening reductions or matmuls — and fuses them into a single Triton kernel. It also fuses simple reductions (RMSNorm, softmax) via its `reduce` lowering. Epilogue fusion (bias after GEMM) uses the CUTLASS/CuTe-DSL backend.

**Where torch.compile falls short.** Cross-layer fusion (e.g., fusing RMSNorm with RoPE when they're two separate `nn.Module` calls) requires the shapes and operations to be visible in a single FX graph. Dynamic shapes, graph breaks, or separate module boundaries can prevent this. Hand-written Triton kernels bypass this — you write exactly the fusion you want.

**The 2026 frontier — ClusterFusion.** ClusterFusion++ (arxiv 2604.23553) fuses an entire decoder layer (RMSNorm → Attention → Add → RMSNorm → MLP) into a single GPU cluster-level kernel. This eliminates HBM writes between every operator. The cost: 3–5× increase in register pressure and SMEM usage, requiring careful tile size selection. This is research-level; the project this week is the more approachable per-op fusion.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | profiling-the-fusion-landscape | Profile LLaMA decoder; build the op-level bandwidth budget |
| 02 | when-to-fuse | Register pressure model; Amdahl limits; the roofline for fusion |
| 03 | triton-swiglu-fusion | Fused SiLU(gate) * up in one Triton kernel |
| 04 | fused-residual-add-norm | RMSNorm + residual add — the most common unfused pair |
| 05 | gemm-epilogue-fusion | Bias + activation inside CUTLASS epilogue |
| 06 | cross-op-fusion-limits | What prevents compile from fusing; what to do about it |
| 07 | clusterfusion-awareness | What full-layer fusion looks like; when it's worth pursuing |

### 01 — `profiling-the-fusion-landscape`

**Build a per-op bandwidth budget.** Before writing a single fused kernel, you need to know where the time is going. The approach:

1. Profile with `torch.profiler` at op granularity (not just layer granularity).
2. For each op: measure wall time + bytes moved (from `ncu --set full`).
3. Compute achieved bandwidth: `bytes / time`.
4. Compare to GPU peak HBM bandwidth (A100: ~2 TB/s; H100: ~3.4 TB/s).
5. Sort by `(time × (1 - bandwidth_utilization))` — this is the "amount of time wasted by not being at peak bandwidth."

The ops at the top of this sorted list are your fusion targets.

**Expected findings for LLaMA 7B on A100:**
- RMSNorm + downstream RoPE: 2 separate kernels, together ~8% of decode time, both at ~30% bandwidth utilization → fusing them doubles bandwidth utilization
- SwiGLU (gate × silu(up)): 2 separate kernels, ~6% of decode time → one fused kernel
- Attention: already at 85%+ utilization via FlashAttention → don't touch
- Linear layers: compute-bound (FP16/BF16 matmul at ~70% MFU) → GEMM epilogue, not Triton

**Build steps.** Run `torch.profiler` on 100 decode steps of LLaMA-7B (or a smaller equivalent). Write a script that extracts per-kernel time + bytes-moved and produces the sorted bandwidth budget table. This table is the first section of your report.

### 02 — `when-to-fuse`

**The fusion decision framework.**

**Register pressure.** Each Triton thread has a finite register file (A100: 65,536 registers per SM, shared across warps). A larger fused kernel holds more intermediate values in registers. If the kernel exceeds the register budget, values spill to local memory (a slow DRAM path). Check register usage: `ncu --set full` → "Registers per Thread."

**Shared memory budget.** Fusing a reduction (RMSNorm) with a pointwise op (RoPE) requires the reduction to complete before the pointwise can begin. The reduction's partial results must live somewhere between the two — in shared memory. The H100 has 228KB SMEM per SM, shared between all resident warps.

**Amdahl limit.** If an op takes 3% of total decode time, a 10× speedup from fusion gives you 0.3% total improvement. Don't spend a week on that. Use the bandwidth budget table from Topic 01 to prioritize ops that are both slow *and* bandwidth-bound *and* large enough to matter.

**The roofline for fusion.** A fused kernel's arithmetic intensity is `sum(FLOPs of all fused ops) / (bytes read + bytes written by the fused kernel)`. Fusing two ops where op1 writes an intermediate that op2 reads eliminates that intermediate from HBM. The arithmetic intensity of the fused kernel is higher → it sits higher on the roofline → it's faster.

### 03 — `triton-swiglu-fusion`

**SwiGLU (used in LLaMA MLP).** `output = silu(gate) * up`, where `gate` and `up` are two separate linear projections of the same input. The unfused version: two matmuls (compute-bound, fine), then `silu(gate)` (a separate elementwise kernel), then `gate * up` (another elementwise kernel) — two unnecessary HBM round-trips.

**The fused version.** One Triton kernel that reads both `gate` and `up` tiles from HBM, computes `silu(gate) * up` in registers, writes the result. One read pass, one write pass.

```python
@triton.jit
def fused_swiglu(gate_ptr, up_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)
    # SiLU: x * sigmoid(x)
    gate_out = gate * tl.sigmoid(gate)
    output = gate_out * up
    tl.store(output_ptr + offsets, output, mask=mask)
```

**Build steps.** Write this kernel. Benchmark vs eager (`silu(gate) * up`) and vs `torch.compile`. Measure: achieved bandwidth (GB/s), vs theoretical peak. Compare to Liger-Kernel's `LigerSwiGLUMLP` — understand every difference.

### 04 — `fused-residual-add-norm`

**The pattern.** Every transformer layer ends with `output = residual + x` followed by `next_input = rms_norm(output)`. Unfused: residual add (elementwise, bandwidth-bound) writes to HBM; RMSNorm reads from HBM. One unnecessary round-trip.

**Fused version.** A single kernel that reads `residual` and `x` once, computes the sum, computes RMSNorm in a single pass (using the online reduction — you wrote this in Level 1), and writes the normalized output. Optionally, write both the normalized output *and* the residual sum (needed for gradient checkpointing).

**This is the Level 1 project extended.** In Level 1 you fused RMSNorm + RoPE. This week you fuse residual_add + RMSNorm, which is different: the residual add doesn't have the RoPE's position-dependency, but it does have the two-input read pattern.

**Build steps.** Write the kernel. Benchmark vs unfused vs `torch.compile`. Compare to Liger-Kernel's `LigerFusedLinearCrossEntropyLoss` for the cross-entropy equivalent pattern (same principle, different ops).

### 05 — `gemm-epilogue-fusion`

**What can go in a GEMM epilogue.** A CUTLASS / CuTe-DSL GEMM epilogue runs immediately after the matmul, while the output tile is still in registers (before writing to HBM). Operations that fit in an epilogue:
- Bias add (`C += bias`)
- Activation (`C = gelu(C)`, `C = relu(C)`)
- Output scaling (`C = C * scale` — used for FP8 requantization)
- ReLU clamp

Operations that don't fit: anything that requires all output tiles to be computed before starting (e.g., softmax over the output — that needs the global max).

**How to add an epilogue in CuTe-DSL:**
```python
# Build an epilogue visitor tree
epilogue = LinearCombination(
    alpha=1.0,
    beta=0.0,
    bias_tma=bias_descriptor,
    activation=GELUActivation()
)
# Pass to the GEMM kernel
gemm_kernel = make_gemm_kernel(mma, epilogue)
```

**Build steps.** Take the BF16 GEMM from Level 4. Add a bias + GELU epilogue. Benchmark vs unfused (GEMM → separate bias add → separate GELU). Measure the memory traffic eliminated.

### 06 — `cross-op-fusion-limits`

**Why `torch.compile` can't always fuse.** The FX graph represents ops in execution order. Inductor fuses pointwise ops eagerly — but only when they're in the same "fusion group." A fusion group boundary appears when:
- There's a reduction op (it forces materialization of the preceding pointwise ops)
- There's a matmul (Inductor doesn't fuse across GEMM boundaries in general)
- There's a graph break (from Level 2)
- Two ops have different tile shapes (different parallelism granularity)

**What you can do.** For ops where `torch.compile` doesn't fuse:
1. Check if a `torch.compiler.disable()` region is causing unnecessary graph breaks.
2. Consider writing a custom Triton kernel and registering it as a `torch.library` custom op — Inductor will treat it as opaque but it'll run your fused code.
3. For GEMM + epilogue: switch to a CUTLASS/CuTe-DSL backend that has the epilogue built in.

**Build steps.** Take the output of your Level 2 graph-break audit. For each remaining fusion gap (places where `torch.compile` failed to fuse ops you expected it to fuse), identify the root cause and apply the most appropriate fix from the three options above.

### 07 — `clusterfusion-awareness`

**What ClusterFusion does.** ClusterFusion++ (arxiv 2604.23553) uses CUDA's cooperative groups to run an entire transformer decoder layer (6–8 op types) in a single GPU cluster-level kernel. The cluster (group of thread blocks sharing NVLink/L2 bandwidth) passes data directly between thread blocks without HBM round-trips.

**Why you're not building it this week.** It requires:
- Cluster-level cooperative groups (H100+ only)
- Careful register pressure management across 6 fused op types
- Custom memory hierarchy analysis (what fits in SMEM vs registers vs L2)
- Weeks of tuning

**What to take away.** Read the ClusterFusion++ paper (or at minimum the abstract + Figure 1). Understand: what intermediate tensors are eliminated? What's the memory traffic reduction claimed? What's the register pressure cost? This is the frontier — understanding it gives you a mental model of where kernel fusion is heading.

**Resources.**
- [ClusterFusion++ — arxiv 2604.23553](https://arxiv.org/html/2604.23553)
- [Deep Kernel Fusion for Transformers — arxiv 2602.11808](https://arxiv.org/html/2602.11808)

## Project this week

```
compiler-and-kernels/
└── fusion/
    ├── bandwidth_budget.py        # op-level profiling + bandwidth table
    ├── fused_swiglu.py            # SwiGLU fusion vs Liger-Kernel
    ├── fused_residual_rms_norm.py # Residual add + RMSNorm
    ├── gemm_with_epilogue.py      # CuTe-DSL GEMM + bias + GELU
    └── reports/
        └── level5-fusion.md      # bandwidth budget table + benchmark results
```

**The report structure:**
1. Bandwidth budget table (per-op time, bytes moved, bandwidth utilization %)
2. Fusion decision for each op (fuse / don't fuse / epilogue — with reason)
3. Benchmark results for 3 fused kernels vs unfused vs torch.compile
4. One paragraph on ClusterFusion: what it achieves and what it costs

## Definition of done

- [ ] You have a per-op bandwidth budget table for a LLaMA decoder block.
- [ ] You have benchmark results for at least 3 fused kernels.
- [ ] You can state the register pressure limit of your fused kernels (`ncu` → "Registers per Thread").
- [ ] You can correctly predict whether `torch.compile` will fuse two ops given their position in the FX graph.
- [ ] `reports/level5-fusion.md` is written.

## Resources

- **Liger-Kernel** — [github.com/linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel). Source for SwiGLU, RMSNorm, RoPE.
- **ClusterFusion++** — [arxiv.org/abs/2604.23553](https://arxiv.org/html/2604.23553).
- **Deep Kernel Fusion for Transformers** — [arxiv.org/abs/2602.11808](https://arxiv.org/html/2602.11808).
- **vLLM MoE kernel features** — [docs.vllm.ai/en/latest/design/moe_kernel_features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/). Production fusion decisions.
- **Dissecting FlashInfer** — [ydnyshhh.github.io/posts/flash_infer](https://ydnyshhh.github.io/posts/flash_infer/). Fusion decisions in the attention layer.

## What you'll be able to do after this week

> Profile a transformer's full kernel fusion landscape, identify bandwidth-bound fusion opportunities, implement hand-fused Triton kernels that outperform torch.compile on reduction-heavy ops, and add GEMM epilogues in CuTe-DSL. Know precisely when hand-fusion beats automatic fusion and when it doesn't.
