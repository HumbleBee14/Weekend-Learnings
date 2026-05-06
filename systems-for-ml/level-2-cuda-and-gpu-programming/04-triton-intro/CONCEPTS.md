# 04 — Triton Intro

## What Triton is

A Python DSL that compiles to PTX. You write kernel logic in Python with NumPy-like operations on *tiles* (blocks of data), and Triton handles the low-level CUDA-isms: register allocation, shared memory layout, vectorization, scheduling, and — as of Triton 3.2 — automatic warp specialization.

Why it matters: serving engines actually use Triton. vLLM has Triton MoE kernels and the new Triton attention backend. SGLang uses Triton extensively. FlashInfer uses Triton for JIT attention kernel generation. Liger-Kernel ships fused RMSNorm/RoPE/SwiGLU/CrossEntropy in Triton. **Triton is what production teams write when they need a custom kernel and don't want to commit to CUDA C++.**

## Triton vs CUDA C++ — the right mental shift

CUDA C++: you think "one thread per output element."
Triton: you think "one program (block) per output tile."

```
              CUDA C++                       Triton
              ────────                       ──────
              one thread = one element       one program = one tile (e.g., 128 elements)
              you manage threads             compiler manages threads
              you allocate shared memory     compiler allocates as needed
              you handle bank conflicts      compiler handles via swizzling
              you write warp shuffles        compiler emits them when reducing
              you specify block size         you specify TILE_SIZE
```

The granularity is bigger. You operate on tiles of data, not individual elements. The compiler maps your tile-level operations to threads, warps, and SMs.

## A first Triton kernel — vector add

```python
import triton
import triton.language as tl

@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Which program (= block) am I?
    pid = tl.program_id(axis=0)

    # Which elements does this program own? A range of BLOCK_SIZE consecutive offsets.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n   # boundary mask

    # Load tiles, compute, store
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


# Launch:
n = 1 << 20
grid = (triton.cdiv(n, 1024),)   # ceil(n / BLOCK_SIZE)
vector_add[grid](a, b, c, n, BLOCK_SIZE=1024)
```

Compare to the CUDA C++ vector_add from Topic 2:
- No `threadIdx.x` — you operate on the whole block's worth of data at once.
- No explicit kernel launch syntax — just `kernel[grid](args, ...)`.
- The `mask` parameter on `tl.load`/`tl.store` handles boundary conditions cleanly.
- `BLOCK_SIZE: tl.constexpr` declares it as a compile-time constant — Triton specializes the kernel for that value.

## Triton matmul — the canonical tutorial

The Triton matmul tutorial ([01-vector-add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html), [03-matrix-multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)) is the canonical reference. ~80 lines. Hits ~95% of cuBLAS on Ampere — same as Boehm's Step 7 in CUDA C++ (which was ~300 lines).

The structure:

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ID: each program computes one BLOCK_M × BLOCK_N tile of C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets into A and B
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer arithmetic to get tile pointers
    A_tile_ptr = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_tile_ptr = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Accumulator in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_tile_ptr)              # BLOCK_M × BLOCK_K tile
        b = tl.load(B_tile_ptr)              # BLOCK_K × BLOCK_N tile
        acc += tl.dot(a, b)                  # tensor-core matmul
        A_tile_ptr += BLOCK_K * stride_ak    # advance pointers
        B_tile_ptr += BLOCK_K * stride_bk

    # Write output tile
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_tile_ptr = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(C_tile_ptr, acc, mask=...)
```

`tl.dot(a, b)` is the magic: it dispatches to tensor cores (HMMA, WGMMA, tcgen05 — whichever the target supports). One line replaces ~50 lines of CUTLASS.

## Autotune

Real Triton kernels use `@triton.autotune` to pick `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages` from a list of configs:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
        # ... typically 5-15 configs
    ],
    key=["M", "N", "K"],   # cache config per (M, N, K) shape
)
@triton.jit
def matmul_kernel(...):
    ...
```

First time the kernel runs at a new shape, autotune benchmarks every config and caches the winner. Subsequent calls at the same shape use the cached choice.

This is why Triton matmul often beats hand-tuned CUDA: the autotuner finds better configs than humans do, and finds them automatically per shape.

## What changed in 2025–2026

- **Triton 3.2** (with PyTorch 2.6, late 2025) — automatic warp specialization landed. The compiler partitions threads into producer warps (TMA loads) and consumer warps (MMA compute) without you writing it. PR [#5622](https://github.com/triton-lang/triton/pull/5622) is the merge. Targeted FlashAttention and FP8 row-wise GEMM, +10–15% on H100.
- **Triton 3.3** (early 2026) — warp specialization requires `num_warps ≥ 4`; ragged TMA support (huge for variable-length attention); MMAv5 pipelining for Blackwell.
- **Tawa pass** ([arXiv 2510.14719](https://arxiv.org/abs/2510.14719)) — the academic paper describing the upstreamed warp-spec transformation. Introduces "async references" (Aref) so the compiler can reason about producer/consumer dataflow.
- **Triton on AMD ROCm** — 2026 Triton runs on MI300X via HIP. Same `.py` files; the autotuner picks different optimal configs per hardware.
- **TMA in Triton** — `tl.make_block_ptr` + `experimental_descriptor_load` is the current path on Hopper+. Ragged TMA (3.3) is the new stable variant.

## What production teams use Triton for in 2026

- **vLLM**:
  - PagedAttention kernels
  - Fused MoE expert dispatch + GEMM (`fused_experts`)
  - RMSNorm + RoPE fusions
  - The new "Triton Attention Backend" ([Mar 2026 deep-dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html))
- **SGLang**: heavy Triton use; their fused MoE benchmarks ([github](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md)) are the canonical reference.
- **FlashInfer**: Triton for JIT attention kernel generation (compile per dtype/head_dim/mask combo at runtime, cache).
- **Liger-Kernel** (LinkedIn, [arXiv 2410.10989](https://arxiv.org/abs/2410.10989)): production fused Triton kernels for RMSNorm, RoPE, SwiGLU, GeGLU, CrossEntropy. +20% throughput, -60% memory in HuggingFace training pipelines.
- **The Anatomy of a Triton Attention Kernel** ([arXiv 2511.11581](https://arxiv.org/abs/2511.11581), Nov 2025): builds a paged-attention kernel purely in Triton that hits 105% of SOTA on both NVIDIA and AMD.

## Pitfalls

1. **`tl.constexpr` confusion.** Anything declared `tl.constexpr` is compile-time, baked into the kernel. Changing it requires recompile. Tile sizes, num_warps, num_stages — all constexpr. Runtime values (M, N, K, pointers) are not.
2. **`mask` not used → segfault.** Without `mask=offsets < n`, threads load past the end of the array. Same as CUDA's bounds check.
3. **Non-power-of-2 BLOCK sizes.** Triton requires power-of-2 tile sizes for many ops. `BLOCK_M=200` won't compile.
4. **First call is slow.** That's autotune + JIT compilation. Cache warms up after one run.
5. **Forgetting `key=` in autotune.** Without it, autotune picks once and uses that config for all shapes. With `key=["M", "N", "K"]`, it caches per-shape.
6. **Triton ≥3.3 warp-spec needs num_warps ≥ 4.** Old code with `num_warps=2` errors out cleanly with a message.

## Why Triton beats your hand-CUDA matmul

- **Autotuner finds better tile configs** than you guessed.
- **Compiler handles tensor-core selection** (HMMA on T4, WGMMA on H100, tcgen05 on B200) — same source, different output.
- **Compiler handles SMEM swizzling** to avoid bank conflicts.
- **Triton 3.2+ adds warp specialization** automatically when applicable.
- **Triton can target AMD ROCm** with the same source. CUDA C++ can't.

The lesson: **for ML kernels in 2026, write Triton first**. Drop to CUDA C++ only when Triton's abstractions don't fit your problem (very rare in LLM serving).

## References

- **Triton tutorials index** — https://triton-lang.org/main/getting-started/tutorials/
- **Persistent matmul tutorial (09)** — https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- **Triton repo** — https://github.com/triton-lang/triton
- **PyTorch — Enabling Warp Specialization in Triton** — https://pytorch.org/blog/warp-specialization/
- **Tawa paper** — https://arxiv.org/abs/2510.14719
- **Triton PR #5622 — automatic warp spec** — https://github.com/triton-lang/triton/pull/5622
- **Ian Barber — How does Triton do Warp Spec?** — https://ianbarber.blog/2025/05/09/how-does-triton-do-warp-spec/
- **The Anatomy of a Triton Attention Kernel** — https://arxiv.org/abs/2511.11581
- **vLLM Triton Attention Backend Deep Dive** — https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html
- **Liger Kernel** — https://github.com/linkedin/Liger-Kernel + https://arxiv.org/abs/2410.10989
- **rkinas/triton-resources (curated meta-list)** — https://github.com/rkinas/triton-resources
