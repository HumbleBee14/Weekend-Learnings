# 09 — Naive KV Cache

## What the KV cache is

Every transformer attention layer computes `softmax(Q · K^T / sqrt(d)) · V`. During autoregressive generation, the K and V tensors accumulate token by token:

```
After token 1:     K = [k1],         V = [v1]
After token 2:     K = [k1, k2],     V = [v1, v2]
After token 3:     K = [k1, k2, k3], V = [v1, v2, v3]
...
After token N:     K = [k1..kN],     V = [v1..vN]
```

If we recomputed K and V every step, we'd do `O(N^2)` work for an N-token sequence. Wasteful — `k_i` and `v_i` don't change after token `i` is processed.

The KV cache: store K and V tensors per layer per request. Each new token only computes `k_new` and `v_new`, appends them, and runs attention against the full cache.

This is why decode is fast at all. Without KV caching, every decode step would be O(N^2). With it, every step is O(N).

## What "naive" means here

The simplest KV cache: a contiguous tensor per layer per request, sized to `max_seq_len`.

```python
# For each layer, for each request in the batch:
k_cache = torch.zeros(max_batch, max_seq_len, n_heads, head_dim, ...)
v_cache = torch.zeros(max_batch, max_seq_len, n_heads, head_dim, ...)

# Generate token at position t:
k_cache[req_idx, t] = k_new
v_cache[req_idx, t] = v_new
```

Simple. It works. But it has problems that motivate paged KV (Topic 10).

## The problems with contiguous KV cache

### 1. Memory waste from over-allocation

If `max_seq_len=8192` and a request only generates 200 tokens, you've reserved 8000 token slots that go unused. Per request. Per layer.

For a 7B model with 32 layers, hidden_size=4096, that's:

```
Per request: 8192 tokens × 32 layers × 2 (K and V) × 4096 hidden × 2 bytes (FP16)
            = 4.3 GB per request
```

With batch size 8 → 35 GB just for KV cache. Most of it unused.

### 2. Internal fragmentation

If you batch 8 requests but they finish at different times, the slots they freed are *interspersed* with active requests' slots. You can't easily compact the cache without breaking attention indexing.

### 3. No prefix sharing across requests

Request A: "You are a helpful assistant. Translate: 'Hello'"
Request B: "You are a helpful assistant. Translate: 'World'"

Both start with the same 8-token prefix. With contiguous KV, you compute that prefix's K and V *twice* — once per request. Wasteful.

### 4. Request length must be known up-front (or capped)

If you set `max_seq_len=2048` and a request needs 4000 tokens, it crashes. If you set 8192 to be safe, you waste memory per request as in (1).

## When naive is fine

Build it once to feel the pain. The pain is the motivation for Topic 10's paged KV.

If you only have one user at a time and bounded sequence length, naive works. Most production has neither constraint.

## What you'll build

A naive KV cache integrated with your `mini-serve` from Level 1. Steps:

1. Add KV cache tensors to your model's forward pass — one tensor per layer, sized for max sequences and max length
2. On the first token (prefill), populate the cache
3. On subsequent tokens (decode), append to the cache and run attention against the populated portion
4. Track per-request positions (where each request is in its cache slot)

Measure: throughput vs the no-cache baseline. Confirm you're getting the O(N) decode that motivates this whole thing.

Then deliberately stress-test it:

- Mixed-length requests in one batch (50-token + 5000-token) → see padding waste
- 100 concurrent requests with `max_seq_len=8192` → run out of memory
- Two requests with the same long prefix → confirm you're recomputing it

Each pain point is a fix in Topics 10-11.

## The interface that doesn't change

In Level 4, the KV cache implementation changes (naive → paged → eviction policies). The *interface* doesn't:

```python
# Inside each attention layer
def attention_with_kv_cache(self, x, kv_cache, position):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    
    # Append k, v to the cache at `position`
    kv_cache.append(layer_id=self.layer_id, position=position, k=k, v=v)
    
    # Read the full populated cache
    full_k, full_v = kv_cache.read(layer_id=self.layer_id, up_to=position + 1)
    
    return scaled_dot_product_attention(q, full_k, full_v)
```

`kv_cache.append` and `kv_cache.read` change implementation between contiguous and paged. The model's forward doesn't.

## Pitfalls

1. **Forgetting to mask the unused portion.** When you compute attention against `k_cache[:, :max_seq_len]`, the unused (zero) slots produce attention weights too. Mask them out.
2. **Confusing token position with cache slot.** Token position is where in the sequence we are; cache slot is where it lives in memory. They're the same in naive, different in paged.
3. **Forgetting layers are independent.** Each transformer layer has its own K and V. Don't share the cache across layers.
4. **Pre-allocating too much.** Naive's biggest sin. Reserve only what you need; better, switch to paged.
5. **Mixing FP16/BF16 KV cache with FP8 model weights.** The KV cache can be in lower precision than the model (Topic 12 covers KV cache compression). State the precision explicitly.

## What you walk away with

- A working naive KV cache integrated with `mini-serve`
- Concrete pain numbers on the four problems above
- The motivation to read Topic 10 (paged KV cache)

## References

- The KV cache concept (Sasha Rush) — https://srush.github.io/llama2.html
- vLLM PagedAttention paper — https://arxiv.org/abs/2309.06180 (Topic 10's main reference)
- HuggingFace KV cache docs — https://huggingface.co/docs/transformers/main/en/kv_cache
