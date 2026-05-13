# 08 — Backward pass survey

> Prereq: sub-module 03 (you understand FA2 forward). Hardware: any GPU optional; this sub-module is mostly reading.

The level so far is inference-shaped: FA2/3/4 forward, FlexAttention forward, FlashInfer (which is inference-only). Training kernels are materially different. This sub-module is short by design — you survey what's different and where to read more, so you're not blindsided when you need to write or extend a training-time attention kernel.

## What's different from forward

### Two separate backward kernels, not one

Forward stores `(O, L)` where `L = m + log(ℓ)` per query row (~`8*N` bytes in fp32 — tiny compared to the `N²` score matrix you didn't materialize). Backward needs `dQ, dK, dV`.

The split: one kernel computes `dQ`; a different kernel computes `dK, dV`. Why? Parallelism alignment. `dQ` is a sum over KV positions for each query row — natural parallelism is across query rows. `dK, dV` are sums over query positions for each KV row — natural parallelism is across KV rows. You want each kernel's parallel-launch axis to match its accumulation pattern, which means two kernels.

### Recomputation, not storage

Backward needs the score matrix `S` and the probabilities `P` to compute gradients:

```
dV = P^T @ dO                     # P is (N, N)
dP = dO @ V^T                     # (N, N)
dS = P * (dP - rowsum(dP * P))    # (N, N), the softmax-Jacobian product
dQ = dS @ K * (1/sqrt(d))
dK = dS^T @ Q * (1/sqrt(d))
```

FA2 backward **re-computes** `S` and `P` tile by tile from `Q, K, V, L`. The stored `L` is what lets you reconstruct `P = exp(S - m) / ℓ` knowing only `S` and the log-sum-exp per row. You never materialize the `N²` matrix, just like forward.

This is the FA2 backward's central trick. It pays you 2× compute for `O(N²)` memory savings.

### `flash_attn_func` vs `flash_attn_varlen_func`

Two forward+backward signatures in the `flash_attn` Python API:

- `flash_attn_func(q, k, v, ...)` — fixed shape `(B, N, H, D)`.
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)` — ragged. `cu_seqlens_*` are prefix sums of sequence lengths.

Training data is almost always packed (concatenated documents separated by EOS tokens) and uses `varlen`. The backward kernel uses the same `cu_seqlens` arrays to ensure gradients don't bleed across document boundaries. Same online-softmax recursion, ragged-friendly indexing.

### Determinism

Default FA2/3/4 backward is **non-deterministic**. `dK, dV` accumulate via atomic adds across the Q parallelism axis. Atomic order is hardware-dependent, so bf16 reductions in different orders give bit-different results. Numerically the answers agree to ~1e-5, but they are not bit-reproducible.

Pass `deterministic=True` to get a serialized backward. Slower (~1.3–1.5× on H100) and uses more memory (a Q-axis partial-sum buffer). Use it for debugging, regression tests, or when you need bit-exact reproducibility.

### FA4 backward (Mar 2026)

The FA4 paper's biggest backward win: use Blackwell's **2-CTA cooperative MMA** mode to keep `dK, dV` partial sums in **TMEM (Tensor Memory)** between tiles instead of round-tripping through HBM via atomics. Two SMs cooperate on one MMA, each providing half the K/V data; the partial sums live in the shared TMEM region. This is what makes FA4 backward faster than FA3 backward by more than the forward speedup would predict.

You cannot reproduce this in Triton today. It requires inline PTX `tcgen05.mma.cta_group::2`. Level 4 (CuTe-DSL) is the right place to learn the tooling.

### Recompute vs save-for-backward

There's a third path: `save_for_backward(S, P)` — i.e., materialize the score matrix during forward and store it. Some inference-only training-adjacent setups (e.g., teacher-forced distillation with frozen teacher) use this when memory is plentiful and you want to skip the backward recompute. Almost no production training does this — the memory cost defeats the FA purpose.

## What you build

Just `notes.md`. A one-page summary of the five points above, in your own words, with enough context for a teammate who finished sub-module 03 to understand what's different in training.

Optional: read the [Triton in-tree tutorial 06](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) — it includes a backward kernel. The dQ kernel and the dK,dV kernel are both ~100 lines of Triton. If you understood the forward, you can read these.

Optional: pick a `flash_attn_varlen_func` call inside the HuggingFace `transformers` training stack and trace what `cu_seqlens` it constructs. The data loader's packing logic is the other half of "how training attention works."

## Definition of done

- [ ] One-page `notes.md` covering the five points: two-kernel split, recomputation, varlen, determinism, FA4-backward.
- [ ] You can answer: "Why are dQ and dK,dV computed in separate kernels?"
- [ ] You can answer: "What does FA2 store to make the backward recomputation possible?"

## References

- [FA2 paper — arXiv 2205.14135](https://arxiv.org/abs/2205.14135) Section 3.2 (backward).
- [Triton in-tree fused-attention tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) — includes a complete forward+backward implementation in Triton.
- [Aleksa Gordić — ELI5 FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) — good plain-English coverage of the backward.
- [ShivamPR21 — FlashAttention Kernel: Backward Pass (Parallelism)](https://shivampr21.github.io/posts/flash-bwd-pll-14-4-2025-kernelized/).
- [flash-attention `flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py) — the Python boundary, including `varlen` signatures.
- [FA4 paper Section 4 (backward)](https://arxiv.org/abs/2603.05451) — the TMEM and 2-CTA-MMA tricks.
