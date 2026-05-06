# 06 — FlashAttention Walkthrough

## Why this is the big one

FlashAttention is the most-cited GPU kernel of the last 5 years and the reason long-context LLMs are practical. Understanding it ties together everything from this level: the memory hierarchy (Topic 5), tiling and reductions (Topic 2), and the online softmax recursion (also Topic 2).

This topic is *understanding-focused*. You're not going to write FA3 or FA4 yourself — those are infrastructure-grade kernels with thousands of lines. You'll implement a minimal version and read the rest.

## The problem with naive attention

Standard attention:

```
Q = (B, H, N, D)    queries
K = (B, H, N, D)    keys
V = (B, H, N, D)    values

S = Q @ K^T   →   shape (B, H, N, N)        attention scores
P = softmax(S, dim=-1)
O = P @ V     →   shape (B, H, N, D)
```

For sequence length N = 8192, hidden size D = 128, head count H = 32, batch B = 1:
- The intermediate S matrix is `1 × 32 × 8192 × 8192 × 2 bytes` (FP16) = **4 GB**.

That's just *one* layer. A 32-layer transformer would materialize 128 GB of intermediate attention matrices over a forward pass — far more than HBM capacity.

Even when it fits, the bandwidth cost dominates. The S matrix is written to HBM, read back for softmax, written again, read again for the matmul with V. Each round trip is at HBM bandwidth (~3 TB/s on H100). Compute (~1 PFLOPS on tensor cores) is barely used.

Naive attention is HBM-bound by a factor of 5-10×. The compute is sitting there idle.

## The FlashAttention idea

**Don't materialize S. Tile the computation so each tile of S lives only in SMEM, never in HBM.**

```
Naive:        Q, K, V → S (in HBM) → softmax → P (in HBM) → O
                       ^^^^^^^^^^^             ^^^^^^^^^^^
                       these round-trip through HBM

FlashAttention: stream K, V tiles through SMEM, compute partial outputs incrementally
                 keep only Q, the output O, and small auxiliary state in HBM
```

The trick: **online softmax**. Topic 2 introduced this. The recursion lets you compute softmax incrementally as you see more values — no need to first compute the global max and sum.

## The FlashAttention-2 algorithm (the canonical version)

Outer loop is over Q tiles (parallel — different blocks). Inner loop is over K, V tiles (sequential within a block).

```
For each Q-tile (rows of output):
    Load Q_tile into SMEM             # (B_R, D), small
    Initialize O_tile = 0             # accumulator in registers/SMEM
    Initialize m = -inf, ℓ = 0        # running max, running sum

    For each K, V tile chunk:
        Load K_tile, V_tile into SMEM         # (B_C, D), small

        # Compute attention scores for this chunk
        S_ij = Q_tile @ K_tile.T              # (B_R, B_C)

        # Online softmax update
        m_new = max(m, max(S_ij, dim=-1))
        scale = exp(m - m_new)
        ℓ_new = ℓ * scale + sum(exp(S_ij - m_new), dim=-1)
        P_ij = exp(S_ij - m_new)              # local softmax numerator

        # Rescale running output, then add new contribution
        O_tile = O_tile * scale + P_ij @ V_tile

        m, ℓ = m_new, ℓ_new

    # Final normalization
    O_tile = O_tile / ℓ
    Write O_tile to HBM
```

The key observation: **at no point is the full S matrix (or the full P matrix) materialized in HBM**. They live tile-by-tile in SMEM and registers.

ASCII visualization for a single Q tile (rows 0-127) iterating over K tiles:

```
Time →

   Iteration 1:                    Iteration 2:                    Iteration 3:
   K cols 0-127                    K cols 128-255                  K cols 256-383
   ─────────────                   ─────────────                   ─────────────
   Q  ▓▓▓                          Q  ▓▓▓                          Q  ▓▓▓
   K  ████████                     K  ████████                     K  ████████   ← swapped
   V  ████████                     V  ████████                     V  ████████   ← swapped

   S=QK^T: (128, 128) tile           S=QK^T: (128, 128) tile         S=QK^T: (128, 128) tile
   stays in SMEM                     stays in SMEM                   stays in SMEM

   m_local, ℓ_local                  m_local, ℓ_local                m_local, ℓ_local
   update running m, ℓ               update running m, ℓ             update running m, ℓ

   O_partial += P @ V_tile           O_partial = rescale + new       O_partial = rescale + new
   (in registers)                                                   

   ─────────────                   ─────────────                   ─────────────

After all K tiles:
   O_final = O_partial / ℓ_final
   Write to HBM ─────────────────────────────────────────────────────────────→

Total HBM traffic: 1 read of Q, all-reads of K and V, 1 write of O. NEVER S or P.
```

## Why this is fast

Bandwidth-wise:
- **Naive**: HBM traffic ∝ N² (the S and P matrices)
- **FlashAttention**: HBM traffic ∝ N (just Q, K, V, O — no quadratic intermediate)

For N=8192 that's a 8192× reduction in attention-related HBM traffic. Real workloads see 2-10× end-to-end speedup because attention isn't the only thing the layer does.

Compute-wise: the FLOPs are the *same*. FA doesn't compute fewer ops; it just doesn't write the intermediate ops to HBM. The win is purely in memory traffic.

## FA1 → FA2 → FA3 → FA4 progression

**FA1 (May 2022, [arXiv 2205.14135](https://arxiv.org/abs/2205.14135))** — original. Tile-based online softmax. Outer loop was over K (rows of S), which limited parallelism. ~2× over standard attention.

**FA2 (July 2023, [arXiv 2307.08691](https://arxiv.org/abs/2307.08691))** — restructured to put outer loop over Q (parallelize across rows of output), reduced non-matmul FLOPs. ~2× over FA1. **This is the version most curricula teach because it's simplest to understand and runs on Ampere (A100) and Hopper.**

**FA3 (July 2024, [arXiv 2407.08608](https://arxiv.org/abs/2407.08608))** — Hopper-specific. Three innovations:
1. **Warp specialization**: producer warps issue TMA loads; consumer warps issue WGMMA + softmax. They run concurrently.
2. **Inter-warp ping-pong**: interleave matmul and softmax so they hide each other's latency.
3. **FP8 with block quantization + Hadamard rotation**: FP8 attention with controlled error.

Hits 740 TFLOPS BF16 (75% util) and ~1.2 PFLOPS FP8 on H100.

**FA4 (March 2026, [Tri Dao's blog](https://tridao.me/blog/2026/flash4/))** — Blackwell-specific, written entirely in **CuTe-DSL** (Python). Four innovations:
1. **5-stage pipeline** (vs FA3's 2-stage): Load → MMA(S=QK^T) → Softmax → MMA(O+=P@V) → Correction, all overlapping.
2. **Software-emulated `exp`**: Blackwell has 2× more FMA units relative to SFU units. FA4 replaces hardware `exp2` with a polynomial approximation on FMA → softmax stops bottlenecking.
3. **Conditional softmax rescaling**: only rescale O when a *new* running max is observed, not on every tile. Saves ~10× rescaling ops.
4. **Five role-specialized warp groups**: Load, MMA, Softmax (8 warps), Correction (4 warps).

Hits 1605 TFLOPS BF16 on B200 (71% util), 1.3× cuDNN, 2.7× Triton.

## FlashInfer — the production layer above FA

FA optimizes a single attention call. **FlashInfer ([github](https://github.com/flashinfer-ai/flashinfer))** is the production kernel layer that wraps FA for serving. It adds:

- **Three KV layouts**: padded (simple), **ragged** (CSR for variable-length batches), **page-table** (paged-attention as in vLLM)
- **Separate kernels** for prefill / append / decode (FA mostly targets prefill)
- **Cascade attention**: reuse KV across requests sharing prefixes (system prompts, multi-turn)
- **JIT compilation**: kernels compile per (dtype, head_dim, mask, layout) at runtime, then cache

vLLM, SGLang, and TensorRT-LLM all use FlashInfer as their attention backend. When someone says "vLLM's attention," they probably mean FlashInfer dispatching to FA2/FA3/FA4 underneath.

Mental model: **FA = the kernel; FlashInfer = the dispatcher + serving layer**.

## What you should write yourself

A minimal FA2 in Triton. ~50 lines. This is exactly the [Triton fused attention tutorial (06)](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html). Working through it gives you the algorithm in working code on your hardware.

Don't try to write FA3 or FA4. Read them. Specifically:
- [FA3 paper](https://arxiv.org/abs/2407.08608) — sections 3 and 4 only
- [Modal's FA4 reverse-engineering](https://modal.com/blog/reverse-engineer-flash-attention-4) — most accessible FA4 walkthrough
- [Tri Dao's FA4 blog](https://tridao.me/blog/2026/flash4/) — the author's own intuition writeup

## The 200-word writeup test

After reading: write 200 words on *"How does FlashAttention compute attention without materializing the N×N matrix?"*

If you can write it crisply with the online softmax recursion spelled out, you have it. If not, re-read.

## Pitfalls

1. **Confusing FA with sparse attention.** They're different. FA is a *dense* attention kernel with smarter memory access. Sparse attention skips computing positions entirely.
2. **Believing FA changes the math.** It doesn't. Same FLOPs, same outputs (modulo numerical noise from order-of-summation differences). Only the memory access changes.
3. **Picking the wrong FA version.** On Hopper, use FA3. On Ampere, FA2. On Blackwell, FA4. dao-ailab/flash-attention dispatches automatically; don't hardcode.
4. **Trying to learn FA before understanding online softmax.** Topic 2's softmax kernel is a prerequisite. The online recursion is the hard part.
5. **Comparing FA vs PyTorch attention without specifying causal/non-causal.** Causal masking changes the bound (only the lower triangle of S is computed). Half the comparisons online don't say which they ran.

## References

**Papers (in order of importance):**
- FlashAttention-1 — https://arxiv.org/abs/2205.14135 (Section 3 only)
- FlashAttention-2 — https://arxiv.org/abs/2307.08691 (the main one to study)
- FlashAttention-3 — https://arxiv.org/abs/2407.08608 (Hopper)
- FlashAttention-4 — https://arxiv.org/abs/2603.05451 (Blackwell, March 2026)

**Blogs (the way to actually learn it):**
- Tri Dao — FA3 — https://tridao.me/blog/2024/flash3/
- Tri Dao — FA4 — https://tridao.me/blog/2026/flash4/
- Modal — FA4 reverse engineered — https://modal.com/blog/reverse-engineer-flash-attention-4
- Lambda — FA4 overview — https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet
- Together AI — FA4 algorithm + pipelining — https://www.together.ai/blog/flashattention-4
- Colfax Research FA3 — https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/
- Colfax Research FA4 — https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
- PyTorch — FA3 announcement — https://pytorch.org/blog/flashattention-3/
- Ian Barber — Cutie Fly (FA4 walkthrough) — https://ianbarber.blog/2026/03/06/cutie-fly/
- Modal — step-by-step FA2 walkthrough — https://modal.com/blog/flash-attention-article

**Code:**
- Dao-AILab/flash-attention — https://github.com/Dao-AILab/flash-attention
- FlashInfer — https://github.com/flashinfer-ai/flashinfer
- Triton FA2 tutorial — https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

**Production context:**
- The Anatomy of a Triton Attention Kernel — https://arxiv.org/abs/2511.11581 (Triton-only paged attention, 105% of SOTA)
- vLLM Triton attention backend deep dive — https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html
- Dissecting FlashInfer — https://ydnyshhh.github.io/posts/flash_infer/
