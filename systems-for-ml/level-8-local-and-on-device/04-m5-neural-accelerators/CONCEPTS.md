# 04 — M5 Neural Accelerators

## What changed in October 2025

M5 introduced a per-GPU-core matmul unit Apple calls a Neural Accelerator (NA). Every GPU core has one. M5 base has 10 NAs; M5 Pro has 20; M5 Max 40-core has 40. They sit *inside* the GPU core, share its register file, and execute matmuls at higher peak FLOPS than the GPU's general FMA path.

```
M3/M4 GPU core                       M5 GPU core
─────────────                        ─────────────
  ALUs (FMA)                           ALUs (FMA)        ← still here, used for
  texture/sampler                      texture/sampler     non-matmul work
                                       Neural Accelerator ← new: dedicated matmul
                                                            unit, fp16/bf16/int8/fp8
                                                            8x perf vs FMA path
```

Apple's published claim: ~4× peak GPU compute for AI workloads versus M4. Real gains on transformer inference are 30–50% in tok/s once you account for the memory-bound nature of decoding (Topic 01 — most of decode is HBM-bound, not FMA-bound).

## How to target them

NAs are programmable from Metal Shading Language and from MLX. There is no public CUDA-style intrinsic; the entry points are:

- **MLX `mx.fast.matmul`** (and ops built on it: `mx.fast.scaled_dot_product_attention`, quantized matmul, etc.). Auto-targets NAs on M5+ when the dtype and shape qualify.
- **Metal Performance Shaders** (MPS framework, not PyTorch MPS) — `MPSMatrixMultiplication` was updated in macOS 26 to dispatch through NAs.
- **Custom Metal kernels** via the `simdgroup_matrix` types for fp16/bf16. The `simdgroup_matrix_multiply_accumulate` builtin uses NAs when available.

llama.cpp does not yet ship NA-aware kernels. As of mid-2026 there are open PRs but no merge. This is the main reason MLX widens its lead on M5 versus M3/M4.

## Practical numbers

7B 4-bit, batch=1 decode, MLX:

```
M3 Max 40-core            ~230 tok/s   (FMA path)
M4 Max 40-core            ~245 tok/s   (FMA path, slightly faster mem)
M5 Max 40-core            ~310 tok/s   (Neural Accelerators)
```

Prefill (compute-bound) sees larger NA gains:

```
2K-token prefill, 7B 4-bit, MLX
M4 Max                    ~3500 tok/s
M5 Max                    ~5200 tok/s     (~1.5x)
```

The asymmetry is the lesson — NAs help most where you are compute-bound. For decode of small batch sizes, you are HBM-bound and the win is bandwidth-shaped.

## Dtype eligibility

In 2026, NAs accelerate:

- **fp16**, **bf16** matmul — primary targets.
- **int8** matmul — fast path for quantized inference at int8 weights or activations.
- **fp8** (E4M3, E5M2) — added in macOS 26.2 / MLX 0.26+. Not all kernels enable it yet.
- **fp4** — partial. Apple's research path uses block-scaled fp4 with two-level scaling similar to NVFP4 (see Apple's M5 LLM exploration paper).

`mx.fast.matmul` picks the right NA codepath when the dtype and shape qualify; otherwise it falls back to the FMA path.

## Memory bandwidth still matters

The NA is a compute unit. Decoding a 70B 4-bit at batch=1 still reads ~38 GB of weights per token, and that does not get faster with NAs. Topic 01's bandwidth math is unchanged. The NAs help when:

- prefill (compute-bound)
- batch>1 decode (compute density rises)
- MoE (each token only touches active experts; a higher fraction of work is matmul)
- training (always compute-bound during forward+backward)

Quote from Apple's M5 LLM research note: *"Neural Accelerators give the largest end-to-end gains on prefill-heavy and training workloads. Decode at batch size 1 remains memory-bound on all M-series."*

## What this means for code

1. Use `mx.fast.matmul` (and the higher-level `mx.fast.*` ops) instead of writing your own `@` for hot kernels.
2. Prefer fp16 / bf16 arithmetic where safe; int8 if the model is quantized that way; fp8 if hardware and MLX version allow.
3. For prefill performance, larger batch (multiple sequences in one prefill) lets you amortize across NAs better.
4. If you write custom Metal, use `simdgroup_matrix` types — Topic 05 has a sample.

## When you do not see the speedup

- Decode at batch 1 is bandwidth-bound. NAs barely help.
- Op falls back to FMA path because dtype is fp32, or shape is too small (NAs prefer multiple of 8 or 16 along contracting dim).
- Memory pressure causes paging — wall clock dominated by swap, NA idle.
- Rosetta 2 path. Make sure you are running native arm64.

## References

- Apple — exploring LLMs with MLX on M5: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- MLX `mx.fast` API: https://ml-explore.github.io/mlx/build/html/python/fast.html
- Metal `simdgroup_matrix` types: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (search "simdgroup_matrix")
- Metal Performance Shaders matrix multiplication: https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
- M5 announcement details: https://www.apple.com/newsroom/ (M5 chip, October 2025)
