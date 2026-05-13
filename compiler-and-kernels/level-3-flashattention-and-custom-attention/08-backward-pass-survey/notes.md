# notes — backward pass survey

## The five points, in my own words

### 1. Two-kernel split

(fill: why dQ and (dK, dV) are separate kernels)

### 2. Recomputation

(fill: what FA2 stores from forward, what it recomputes in backward, the role of L)

### 3. varlen

(fill: how cu_seqlens enables packed-document training without cross-doc gradient bleed)

### 4. Determinism

(fill: why default is non-deterministic, how deterministic=True works, when to use it)

### 5. FA4 backward

(fill: 2-CTA MMA + TMEM partial sums, why this is Blackwell-only)

## When I will need to actually write a backward kernel

- I'm writing a custom training-time attention variant that FlexAttention can't express. FlexAttention's compiler emits the backward too — for most custom variants I will not need to write a backward kernel by hand.
- I'm porting an inference kernel to support training (e.g., LoRA fine-tuning that includes attention weight gradients). Then the dQ/dK/dV split and recomputation pattern is what I follow.

Otherwise I trust FlashAttention's varlen backward, FlexAttention's auto-generated backward, or PyTorch's SDPA backward — they cover ~99% of training needs.
