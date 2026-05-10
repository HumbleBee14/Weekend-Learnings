# 07 — torch.compile for Inference

## What changed since 2024

- **vLLM V1 enables `torch.compile` by default.** It's no longer experimental.
- **Piecewise CUDA graphs** is the canonical inference compile recipe in 2026.
- **Compile artifact caching** is on by default (`~/.cache/vllm/torch_compile_cache`) — cold start cost amortizes after first run.
- **Model Runner V2** (late 2025) adds piecewise CUDA graphs for **pipeline parallelism**.
- `compiler-and-kernels` Level 2 has the deeper internals; this topic is the practical inference recipe.

## The problem torch.compile solves for inference

Eager PyTorch:
- Each op launches a CUDA kernel
- Each launch has CPU overhead (~5µs for `cudaLaunchKernel`)
- Many small kernels → CPU is the bottleneck on small batches

`torch.compile`:
- Traces the model into an FX graph
- Inductor lowers the graph to fused Triton kernels
- Kernel count drops 3-10×; CPU launch overhead drops proportionally
- Fusion eliminates intermediate writes to HBM

For decode at low batch (where Python+launch overhead is significant), this is a real 1.3-2× win.

## Why naive torch.compile breaks for inference

LLM inference has a problem: **dynamic shapes**.

- Sequence length varies per request
- Batch size varies as requests arrive/finish
- KV cache grows token by token

`torch.compile` recompiles per shape by default. With unbounded shape variation → constant recompilation → slower than eager.

Fixes that have been tried:

1. **Dynamic shape compilation** (`mark_dynamic`) — works for some patterns, hits limitations on attention
2. **Padding to bucket sizes** — compile for batch=1, 2, 4, 8, 16, pad incoming requests to nearest bucket. Memory and compute waste.
3. **Piecewise CUDA graphs** — the 2026 winner. Below.

## Piecewise CUDA graphs — the canonical inference recipe

The key insight: most of the model has *predictable* shapes (token-wise ops, layer norm, projections). Only attention has dynamic shape (KV cache length grows).

So:

```
┌──────────────────────────────────────────────┐
│ Pre-attention (token-wise compute)           │
│   embedding, RMSNorm, Q/K/V projections      │
│   Static shapes: (batch, hidden)             │
│                  ↓                           │
│   CUDA-graph-captured. Replays in 1µs.      │
└──────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│ Attention                                     │
│   QK^T, softmax, @V                          │
│   Dynamic shape: KV cache grows              │
│                  ↓                           │
│   Eager mode. Calls FlashAttention/         │
│   FlashInfer kernel.                         │
└──────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│ Post-attention (token-wise compute)          │
│   Output projection, RMSNorm, MLP            │
│   Static shapes: (batch, hidden)             │
│                  ↓                           │
│   CUDA-graph-captured.                       │
└──────────────────────────────────────────────┘
```

vLLM V1 uses this pattern in production. The pre-attention and post-attention pieces benefit from torch.compile's fusion AND CUDA-graph capture; attention runs eager because of the variable KV length.

## How vLLM exposes this

```bash
# torch.compile is on by default in vLLM V1
vllm serve Qwen/Qwen2.5-1.5B-Instruct

# Inspect the compile cache
ls ~/.cache/vllm/torch_compile_cache/

# Override capture sizes (which batch sizes get CUDA-graph-captured)
vllm serve <model> --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'

# Disable compile (for debugging)
vllm serve <model> --enforce-eager
```

The first request hits the cold-compile path: ~30-60 seconds for a 7B model. Subsequent runs (or warm cache) start in seconds.

## What you'll see in the profile

In Level 3's torch.profiler trace of a vLLM-V1-served model:

- Pre-attention: **one `triton_*` kernel** instead of 5-10 separate ops
- Attention: **flash_fwd_*** kernel, eager
- Post-attention: another fused `triton_*` kernel
- Total kernel count: 3 per layer instead of 15-20

That's the win: kernel-launch overhead × decode steps × layers. For a 32-layer model at batch=1, that's hundreds of thousands of saved kernel launches per second.

## torch.compile gotchas in 2026

1. **Cold-start time is real.** First compile of a 7B model: 30-60 seconds. 70B: 3-5 minutes. Cache it. Use AOTInductor for cold-start-sensitive deployments.

2. **Recompilations on shape change.** If your bucket sizes don't cover the actual shapes you see, recompile cost dominates. Pre-warm with realistic shapes.

3. **Graph breaks.** `torch.compile` can't trace some Python constructs (logging, dynamic control flow on tensor values). Use `TORCH_LOGS=graph_breaks` to find them. For deep diagnosis: `compiler-and-kernels` Level 2 covers `depyf` and graph-break audits.

4. **torch.compile + torchao composability.** Had bugs through PyTorch 2.7 — don't blindly mix quantization and compile until you've tested. Resolved in 2.8.

5. **Logging and side effects break tracing.** Move `print()` and `wandb.log()` outside the compiled region.

## When to use what

```
Workload                                                Recipe
──────────────────────────────────────────────────────────────────────
Production LLM serving (any size)                       vLLM V1 default (piecewise CUDA graph)
Custom serving stack                                    torch.compile + manual CUDA graph capture
Cold-start sensitive (serverless, edge)                 AOTInductor (compile once, ship .so)
Research / iteration                                    Eager. Don't compile until you need to.
```

For the curriculum: when you build `mini-vllm` in Levels 4-5, apply torch.compile to your model. Topics 09-12 (KV cache) are the part where you need to be careful about shapes.

## Pitfalls

1. **Compiling without warmup.** First inference includes JIT cost. Always warm before timing.
2. **Comparing eager throughput to one compile call.** Compile costs amortize over many calls. Measure steady-state.
3. **Using torch.compile on the whole forward when attention has dynamic shapes.** Use the piecewise pattern.
4. **Forgetting that compile's win is biggest at low batch.** At batch=64+, kernels are already big enough that launch overhead is small.
5. **Ignoring cache directory growth.** `~/.cache/vllm/torch_compile_cache` can grow to GB. Clean periodically.

## Connection to compiler-and-kernels Level 2

That track goes deeper:
- Dynamo bytecode tracing internals
- FX graph manipulation
- Reading the generated Triton source
- depyf for graph-break debugging
- The piecewise CUDA graph pattern's actual implementation

This topic is the *user* of those mechanics. If something goes wrong with torch.compile in production, that's where you go.

## References

- vLLM torch.compile design — https://docs.vllm.ai/en/latest/design/v1/torch_compile.html
- vLLM piecewise CUDA graphs (Aug 2025 blog) — https://blog.vllm.ai/2025/08/20/torch-compile.html
- PyTorch torch.compile docs — https://docs.pytorch.org/docs/stable/torch.compiler.html
- AOTInductor for ahead-of-time compile — https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html
