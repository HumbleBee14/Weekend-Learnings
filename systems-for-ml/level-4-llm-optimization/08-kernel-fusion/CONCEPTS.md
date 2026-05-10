# 08 — Kernel Fusion

## Why fusion is the prize

Decode is memory-bound (Level 3, Topic 04). Most of the time is spent reading data from HBM. *Anything* that eliminates an HBM round-trip is a direct speed win.

Fusion = combining multiple operations into a single kernel that reads inputs once and writes outputs once. The intermediate values stay in registers or shared memory, never touching HBM.

Concrete: a typical transformer layer has these operations:

```
input → RMSNorm → Q/K/V projections → Attention → output projection → residual add → RMSNorm → MLP gate → SiLU → MLP up → MLP down → residual add
```

Unfused: 12+ kernel launches, 12+ HBM round-trips for intermediates.
Fused (production-tuned): 4-5 kernel launches, ~5 HBM round-trips. **3× fewer HBM round-trips → 2-3× faster decode**.

## What fusion can and can't combine

**Easy fusions** (pointwise ops back-to-back):
- residual_add + RMSNorm → fused
- silu + multiply → fused (SwiGLU)
- bias + activation → fused (GEMM epilogue)
- rope rotation + key/value writes → fused

**Hard fusions** (cross GEMM/attention boundaries):
- Q/K/V projection + RoPE → harder; requires custom kernel (FlashInfer)
- RMSNorm + Q projection → possible but rare
- Attention + output projection → not really fused; pipeline overlap is the play

**Won't fuse** (would change semantics):
- Operations on different shapes
- Operations that need barriers between them (reductions across whole tensor)

## The 2026 production toolbox

Three tools for fusion, in order of how often they're used:

### 1. torch.compile / Inductor (Topic 07)

Automatically fuses pointwise sequences and simple reductions. Free, works on any PyTorch model. The first thing you reach for.

Limitation: doesn't fuse across GEMM or attention boundaries; doesn't always find the optimal fusion.

### 2. Liger-Kernel — production fused ops (LinkedIn)

Hand-written Triton kernels for the common patterns. Drop-in replacements for PyTorch ops. Used heavily in training; increasingly in inference too:

- `liger_rms_norm` — fused RMSNorm
- `liger_swiglu` — fused gate × silu(up)
- `liger_rope` — fused rotary embedding
- `liger_fused_linear_cross_entropy` — output projection + softmax + loss in one kernel (training)

Win: typically 1.5-3× over torch.compile on the same op. Memory savings can be larger.

### 3. FlashInfer — fused inference attention

Same library you've seen in Level 2 Topic 06. For inference specifically, FlashInfer provides:

- Fused RoPE + attention + output projection (1.6-3.7× bandwidth win)
- Page-table attention (paged KV cache, Topic 10)
- Cascade attention for shared prefixes
- JIT compilation per (dtype, head_dim, mask, layout) combo

The 2026 reality: vLLM, SGLang, and TensorRT-LLM all use FlashInfer underneath. When someone says "vLLM's attention," they usually mean FlashInfer's attention.

### 4. Custom CUTLASS / CuTe-DSL (specialist territory)

For the cases where Liger-Kernel and FlashInfer don't have your specific op. `compiler-and-kernels` Level 4 covers writing your own.

## Concrete example: RMSNorm fusion

Unfused PyTorch:

```python
# 5 kernel launches, 5 HBM round-trips
norm = x.pow(2).mean(-1, keepdim=True)        # write to HBM
norm = norm.add(eps).rsqrt()                  # read, write
x = x * norm                                  # read both, write
x = x * weight                                # read, write
```

Fused (Liger-Kernel):

```python
from liger_kernel.transformers import LigerRMSNorm
norm = LigerRMSNorm(hidden_size, eps=1e-6)
y = norm(x)                                   # 1 kernel launch, 1 HBM read of x, 1 write of y
```

Same math. ~3× faster on H100 because of HBM round-trip elimination.

## Concrete example: SwiGLU fusion

Standard MLP:

```python
gate = self.gate_proj(x)                      # GEMM
up   = self.up_proj(x)                        # GEMM
hidden = F.silu(gate) * up                    # 2 kernels: silu, multiply
out = self.down_proj(hidden)                  # GEMM
```

The 2 small kernels (silu + multiply) are the fusion target.

Fused (Liger-Kernel):

```python
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
hidden = LigerSiLUMulFunction.apply(gate, up)  # one kernel
```

Or via torch.compile: it'll fuse the silu + multiply automatically into a single Triton kernel. The Liger version is often faster because it's hand-tuned, but compile is "good enough" for most cases.

## When custom fusion beats torch.compile

torch.compile finds local fusion opportunities but doesn't know your hardware deeply. Hand-written kernels can:

- Use the right tile sizes for your specific GPU
- Exploit warp specialization (Topic 06 of Level 2)
- Use specialized instructions (TMA, WGMMA, tcgen05)
- Pipeline with surrounding ops in ways the compiler can't see

Rule of thumb: if your profile (Level 3) shows torch.compile-fused kernels at <60% Speed-of-Light, there's room for hand-tuning. If they're at 80%+, leave them alone.

## Pitfalls

1. **Fusing for fusion's sake.** Not every op fusion is a win. If the compiler already handles it well, custom kernels add maintenance burden.
2. **Forgetting torch.compile compatibility.** Custom Triton kernels often don't compose with torch.compile's tracing. Annotate with `@torch.library.custom_op` or wrap properly.
3. **Hand-fusing memory-saturated ops.** Fusion only helps memory-bound ops. A GEMM that's already at 95% compute SOL won't get faster from epilogue fusion.
4. **Over-fusing into one giant kernel.** Big fused kernels have register pressure and shared memory pressure. Past a point, fusion *hurts*. The sweet spot is 3-7 ops per fused kernel.
5. **Skipping the profile-first step.** Always measure where the time goes before deciding what to fuse.

## What you'll do

For your `mini-vllm`:

1. Apply `torch.compile` (Topic 07).
2. Profile (Level 3 tools). Find which ops aren't fused.
3. Replace the standard `RMSNorm` and `SwiGLU` with Liger-Kernel equivalents.
4. Measure delta.

For attention specifically: vLLM/FlashInfer is already fused; use it via vLLM's serving path rather than rebuilding.

## References

- Liger-Kernel — https://github.com/linkedin/Liger-Kernel
- FlashInfer — https://github.com/flashinfer-ai/flashinfer
- FlashInfer NVIDIA blog — https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/
- Horace He's "Making Deep Learning Go Brrrr" (the canonical fusion philosophy) — https://horace.io/brrr_intro.html
- compiler-and-kernels Level 1 (Triton fusion patterns)
- compiler-and-kernels Level 5 (kernel fusion deep dive)
