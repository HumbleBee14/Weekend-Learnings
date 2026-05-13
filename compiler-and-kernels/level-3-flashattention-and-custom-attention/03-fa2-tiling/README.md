# 03 — FA2 tiling: NumPy first, then Triton

> Prereq: sub-modules 01–02. Hardware: T4 (free Colab) is enough; A100 makes the comparison numbers meaningful.

This is the central exercise of the level. Once you have a working FA2 forward, the rest of the level is variations on the theme.

## What FA2 is

FA2 ([Tri Dao, arXiv 2205.14135](https://arxiv.org/abs/2205.14135)) is FlashAttention with the loop order and parallelism that the GPU actually wants:

- **Outer loop: over Q tiles** (parallel; one CUDA program per Q tile).
- **Inner loop: over K/V tiles** (serial within a program).
- **Persistent state in registers/SRAM:** `(m_i, ℓ_i, O_i)` for each row of the Q tile.
- At end of the inner loop, divide `O_i` by `ℓ_i` and write to HBM.

The original FlashAttention-1 had the loop order swapped (outer over KV, inner over Q) which forced more atomic writes and SMEM traffic for `O`. FA2 fixed that. Everything since (FA3, FA4) keeps the FA2 loop order; the changes are about how to pipeline and schedule it.

The tile loop, drawn:

```
                        K, V columns  (stream through SRAM, one tile at a time)
                        ┌────┬────┬────┬────┬────┬────┐
                        │ K0 │ K1 │ K2 │ K3 │ K4 │ K5 │   ← inner loop, serial
                        │ V0 │ V1 │ V2 │ V3 │ V4 │ V5 │     per Q tile
                        └────┴────┴────┴────┴────┴────┘
                          j=0  j=1  j=2  j=3  j=4  j=5
        Q rows
        ┌────┐         ┌──────────────────────────────┐
   i=0  │ Q0 │ ──┐     │  resident in SRAM/registers  │
        ├────┤   │     │       for this program       │
   i=1  │ Q1 │ ──┤     │   state: (m_i, ℓ_i, O_i)     │
        ├────┤   │     │   per row of the Q tile      │
   i=2  │ Q2 │ ──┼──►  │                              │
        ├────┤   │     │   for j in K/V tiles:        │
   i=3  │ Q3 │ ──┤     │      load K_j, V_j           │
        ├────┤   │     │      S = Q_i Kjᵀ             │
   i=4  │ Q4 │ ──┘     │      m_new = max(m_i, max S) │
        └────┘         │      rescale = exp(m_i-m_new)│
        ▲              │      ℓ_i = ℓ_i·rescale + ΣP  │
        │              │      O_i = O_i·rescale + P·V │
   outer loop          │      m_i = m_new             │
   PARALLEL            │   O_i /= ℓ_i ; write O[i:]   │
   (one program        └──────────────────────────────┘
    per Q tile)
```

Q tile and `(m_i, ℓ_i, O_i)` stay resident for the lifetime of one program. K and V tiles stream in and are discarded. That asymmetry is why HBM traffic collapses to `O(N·d)`: each Q row is loaded once, each K/V tile is loaded `N/BLOCK_M` times — but each load is amortized over a full block of Q rows.

## What you build

1. `fa2_numpy.py` — block FA2 in NumPy. Outer loop over Q blocks, inner over KV blocks, maintain `(m, ℓ, O)` per row of the Q block. Verify bit-equal vs `attention_ref` from sub-module 01.
2. `fa2_triton.py` — port to Triton. One program per Q block. Reference: [Triton in-tree tutorial 06](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py).
3. `bench_fa2.py` — `triton.testing.do_bench` against `F.scaled_dot_product_attention`. Print a table.
4. `notes.md` — every pitfall you hit.

Targets:
- NumPy: matches `attention_ref` to `1e-10` in fp64. Mandatory.
- Triton on T4: matches PyTorch SDPA(MATH) to `1e-3` in fp32. Within ~2× of the MATH backend's speed.
- Triton on A100: matches PyTorch SDPA(FLASH) to `5e-3` in bf16. Within 2–3× of FA2's reported speed; matching FA2 exactly is a multi-week project, not a one-day kernel.

## The FA2 algorithm, line by line

Algorithm 1 from the FA2 paper, simplified to forward + no causal mask:

```
Inputs: Q[N,d], K[N,d], V[N,d]
Outputs: O[N,d], L[N]  (L is the log-sum-exp per row, saved for backward)

for i in range(0, N, BLOCK_M):                # outer: Q tiles. PARALLEL.
    Q_i = Q[i:i+BLOCK_M]                      # (BLOCK_M, d), into SRAM
    m_i = -inf * ones(BLOCK_M)                # row max so far
    l_i = zeros(BLOCK_M)                      # row denom so far
    O_i = zeros(BLOCK_M, d)                   # row output so far

    for j in range(0, N, BLOCK_N):            # inner: KV tiles. SERIAL.
        K_j = K[j:j+BLOCK_N]                  # (BLOCK_N, d) into SRAM
        V_j = V[j:j+BLOCK_N]                  # (BLOCK_N, d) into SRAM
        S_ij = Q_i @ K_j.T * (1/sqrt(d))      # (BLOCK_M, BLOCK_N) in registers
        m_new = maximum(m_i, max(S_ij, axis=-1))
        P_ij = exp(S_ij - m_new[:, None])     # (BLOCK_M, BLOCK_N)
        rescale = exp(m_i - m_new)            # (BLOCK_M,)
        l_i = l_i * rescale + sum(P_ij, axis=-1)
        O_i = O_i * rescale[:, None] + P_ij @ V_j   # (BLOCK_M, d)
        m_i = m_new

    O_i = O_i / l_i[:, None]
    L_i = m_i + log(l_i)
    write O[i:i+BLOCK_M] = O_i
    write L[i:i+BLOCK_M] = L_i
```

The whole forward kernel is 15 logical lines. The art is in the Triton transcription: tile sizes, register pressure, `tl.dot`'s accumulator dtype, masking, autotune.

## The rescale step (where you will get stuck)

The line `O_i = O_i * rescale[:, None] + P_ij @ V_j` is the heart. Three things to get right:

1. **Order.** Multiply `O_i` by `rescale` *before* adding `P_ij @ V_j`. If you do `(O_i + P_ij @ V_j) * rescale` you have rescaled the *new* tile's contribution, which is wrong. The unit test catches this; the bug is also visually obvious in the output (output looks "smeared" along Q dim).
2. **Broadcast.** `rescale` is `(BLOCK_M,)`; `O_i` is `(BLOCK_M, d)`. The broadcast dim is the head dim. Easy to fat-finger in Triton because `tl.broadcast_to` is implicit and silent.
3. **Numerical stability.** `m_i` starts at `-inf`. `exp(-inf - m_new) = 0`, which means on the first tile the rescale wipes the zero-initialized `O_i` and `l_i`. This is correct and intended — but if you initialize `m_i` to something finite "to be safe", you will add garbage on the first tile.

## The Triton transcription

You have written tiled matmul (Level 1 sub-module 04). FA2 is two matmuls (`QK^T` and `PV`) with a softmax in between, with the wrinkle that the inner-loop accumulator state is `(m, ℓ, O)` instead of just `acc`.

Skeleton:

```python
@triton.jit
def fa2_fwd_kernel(
    Q, K, V, O,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,  # and so on for K, V, O
    N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # which Q block
    pid_bh = tl.program_id(1)  # which (batch, head)
    # offsets...
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + ...)  # (BLOCK_M, D), masked

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    O_i = tl.zeros([BLOCK_M, D], tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + ...)  # (BLOCK_N, D)
        v = tl.load(V + ...)  # (BLOCK_N, D)

        s = tl.dot(q, tl.trans(k)) * sm_scale   # (BLOCK_M, BLOCK_N)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        rescale = tl.exp(m_i - m_new)

        l_i = l_i * rescale + tl.sum(p, axis=1)
        O_i = O_i * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    O_i = O_i / l_i[:, None]
    tl.store(O + ..., O_i.to(O.dtype.element_ty))
```

You will write a full version with proper masking on boundary tiles and pointer arithmetic. The Triton tutorial 06 is the canonical reference.

**Optional follow-up: `tl.make_tensor_descriptor`.** On Hopper+, descriptor-based loads lower to TMA and unlock async pipelining. On T4 they do nothing perf-wise. We flag the lines but do not require them for the T4 path.

## Benchmark table (your `bench_fa2.py` should produce something like)

A100, N=4096, d=64, B*H=32, bf16:

| Impl | ms/iter | TFLOPs/s | vs SDPA(FLASH) |
|---|---|---|---|
| F.sdpa MATH | x | y | baseline |
| F.sdpa FLASH (FA2) | x | y | 1.00x |
| Your Triton FA2 | x | y | 0.3–0.5x typical first try |

If you land within 2× of FA2 on the first try you are doing great. The gap is mostly: no warp specialization (next sub-module), suboptimal tile size, no autotune over the inner loop strides.

## Definition of done

- [ ] `fa2_numpy.py` matches `attention_ref` to `1e-10` in fp64.
- [ ] `fa2_triton.py` matches `F.scaled_dot_product_attention` to bf16 tolerance.
- [ ] `bench_fa2.py` prints the table above for your hardware.
- [ ] `notes.md` has every pitfall you hit, especially the rescale-order one (you will hit it).

## References

- [FA2 paper — arXiv 2205.14135](https://arxiv.org/abs/2205.14135), Section 3 and Algorithm 1.
- [Triton in-tree fused-attention tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py).
- [Anatomy of a Triton Attention Kernel — arXiv 2511.11581](https://arxiv.org/abs/2511.11581) Section 3.
- [Alex Dremov — Understanding Flash Attention: writing the kernel in Triton](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/).
- [Modal — What is Flash Attention?](https://modal.com/blog/flash-attention-article).
