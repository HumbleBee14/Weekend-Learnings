# 06 — FlexAttention: custom attention you can actually ship

> Prereq: sub-modules 01–03. Hardware: A100 ideal; works on T4 with smaller shapes.

This is the highest-value applied sub-module of the level. FlexAttention is how you ship a custom attention variant in a real project without writing a Triton kernel. Three Python lines and `torch.compile` builds you a fused kernel with block-sparse skipping that matches hand-Triton within ~10% on typical variants.

## The API

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def score_mod(score, b, h, q_idx, kv_idx):
    # score: scalar, the Q[b,h,q_idx] @ K[b,h,kv_idx] dot product, post-scale.
    # b, h, q_idx, kv_idx: integer indices.
    # return: modified score (still pre-softmax).
    return score

def mask_mod(b, h, q_idx, kv_idx):
    # return: True if this (q, kv) position should attend, False to mask to -inf.
    return q_idx >= kv_idx

block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N)
out = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
```

Two things to internalize:

1. **`score_mod` is per-element, runs pre-softmax.** Use for ALiBi, RoPE biases, soft-capping, custom scales.
2. **`mask_mod` is per-element, returns bool.** Use for sparsity patterns. The compiler builds a `BlockMask` (default block 128×128, 256×128 on Blackwell) and the kernel skips fully-masked blocks. Speedup is proportional to sparsity.

`score_mod` does not get block-sparsity skipping (every position is still touched). `mask_mod` does. Use `mask_mod` whenever the mod is `score = score if condition else -inf`.

## How it compiles

`flex_attention` traces `score_mod` and `mask_mod` as FX graphs. With `torch.compile`, Inductor lowers them into Triton tensor operations and **inlines them into the attention kernel body**:
- `score_mod` is inlined into the softmax warp (pointwise, register-backed).
- `mask_mod` is consulted at the block level to skip entire 128×128 tiles, and at the element level inside partial blocks to mask to -inf.

With `kernel_options={"BACKEND": "FLASH"}` on Hopper/Blackwell, the lowering target is FA4's CuTeDSL kernel instead of Inductor Triton. Same `score_mod`, same `mask_mod`, different backend. The PyTorch + FA4 blog (Mar 2026) reports these speedups vs vanilla FA2 baseline:

| Variant | GB200 fwd | GB200 bwd | H200 fwd | H200 bwd |
|---|---|---|---|---|
| ALiBi | 1.2–2.1× | 1.9–2.9× | 1.30–1.54× | 1.36–1.65× |
| Document mask | up to 2.7× | up to 3× | 1.41–1.89× | 1.48–2.01× |
| Sliding window | 1.4–2.1× | 1.8–2.2× | 1.45–1.65× | 1.35–1.52× |

The key claim: **a Python `score_mod` is not a performance compromise.** It compiles into the same fused kernel that ships the SOTA.

## What you build

Four scripts.

### `alibi.py` — ALiBi via `score_mod`

```python
alibi_slopes = torch.tensor([2 ** (-(2 ** -(i / 8))) for i in range(1, H+1)], device="cuda")

def alibi_score(score, b, h, q_idx, kv_idx):
    return score - alibi_slopes[h] * (q_idx - kv_idx).abs()
```

Bit-verify (within bf16 tolerance) against a NumPy reference where you build the explicit ALiBi bias matrix and add it before softmax.

### `sliding_window.py` — sliding-window causal via `mask_mod`

```python
WINDOW = 1024

def sliding_causal(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= WINDOW)

block_mask = create_block_mask(sliding_causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
```

Benchmark vs full causal SDPA at N ∈ {2048, 4096, 8192}. Expect 3–8× speedup on A100 at N=8192, W=1024 (block sparsity = ~88%).

Print the `block_mask.sparsity()` value alongside the speedup — they should track each other.

### `document_mask.py` — packed multi-document batching

```python
doc_ids = build_doc_ids(seq_lens)  # (N,) int tensor

def doc_mask(b, h, q_idx, kv_idx):
    return doc_ids[q_idx] == doc_ids[kv_idx]

block_mask = create_block_mask(doc_mask, ...)
```

This is the production use case for fine-tuning data loaders that pack variable-length samples into fixed-length sequences. Each token must attend only within its own document. With four 2048-token documents packed into N=8192, block sparsity is ~75%.

### `read_emitted_kernel.py` — what did the compiler produce?

Run each of the three variants under:

```python
import os
os.environ["TORCH_LOGS"] = "output_code"
# ... compile and run flex_attention
```

Or use `depyf` from Level 2 to dump the Inductor-generated Triton to a file. Open the generated kernel, find:

- The line(s) corresponding to your `score_mod` inlined into the softmax. For ALiBi, you should see something like `score -= alibi_slope * abs(q_idx - kv_idx)` woven into the inner loop.
- The block-skip logic — Inductor emits `if block_mask_indices[m_block, n_block] == 0: continue` or equivalent.
- The fallback path for partial blocks where some elements are masked.

Write three sentences per variant in `notes.md` summarizing what got inlined.

## Don't make these mistakes

- **Benchmarking the uncompiled path.** `flex_attention(...)` without `torch.compile` is slow. Always `torch.compile(flex_attention, dynamic=False)` or use it inside a `torch.compile`-wrapped model.
- **Recreating `BlockMask` every forward.** `create_block_mask` runs a kernel; do it once per shape, cache it. Static shapes are fine; dynamic shapes need `BlockMask` recomputation.
- **`score_mod` for masking.** A `score_mod` that returns `-inf` works but you lose block sparsity. Use `mask_mod` for masking; use `score_mod` only for additive biases.
- **Closure variables that change.** Captured tensors (like `alibi_slopes`) can change values without recompile. Captured *scalars* (like a Python int) recompile when they change. Prefer tensor closure.
- **Forgetting that backward also gets compiled.** FlexAttention emits a fused backward too. Most of the speedup numbers above include the backward pass. The FlexAttention paper notes: 0.86×–1.05× backward parity with FA2 on supported variants — i.e., almost-free differentiability.

## Definition of done

- [ ] `alibi.py` matches a NumPy ALiBi reference to bf16 tolerance, with a benchmark vs SDPA.
- [ ] `sliding_window.py` runs at N=8192, W=1024, reports both `block_mask.sparsity()` and the measured speedup, and they roughly agree.
- [ ] `document_mask.py` runs with 4 packed documents and produces correct output (cross-document attention scores should be exactly the rows of P that are zero where doc_id differs).
- [ ] `read_emitted_kernel.py` produces an output kernel file for each variant; you wrote three sentences per variant in `notes.md`.

## References

- [PyTorch — FlexAttention launch blog (Aug 2024)](https://pytorch.org/blog/flexattention/) — the original API + examples.
- [PyTorch — FlexAttention + FlashAttention-4 (Mar 2026)](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) — the FA4 backend, with benchmark tables.
- [FlexAttention paper — arXiv 2412.05496](https://arxiv.org/abs/2412.05496).
- [PyTorch source — `torch/nn/attention/flex_attention.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py).
- [attention-gym](https://github.com/pytorch-labs/attention-gym) — reference implementations of dozens of variants. Steal liberally.
- [Colfax — A User's Guide to FlexAttention in FlashAttention CuTeDSL](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/).
- [Jonathan Chang — vLLM from scratch with FlexAttention](https://jonathanc.net/blog/vllm-flex-attention-from-scratch) — for the prefill/decode-shaped use case.
