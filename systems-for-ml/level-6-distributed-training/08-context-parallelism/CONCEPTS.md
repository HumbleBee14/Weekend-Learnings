# 08 — Context Parallelism

CP splits the *sequence dimension* across devices. Used for long-context training (≥32K, often ≥1M). Ring Attention or Striped Attention is the underlying algorithm.

## What CP does

Each device holds a contiguous slice of the sequence's K/V (and a slice of Q). Attention is computed by passing K/V slices around a ring while Q stays put. Each device computes partial attention scores against the slices it sees, then accumulates.

```
sequence of 8K tokens, CP=4

device 0: tokens [0  ..2047]    holds Q[0..2047], K[0..2047], V[0..2047]
device 1: tokens [2048..4095]   holds Q[...],     K[...],     V[...]
device 2: tokens [4096..6143]
device 3: tokens [6144..8191]

Each step around the ring:
  device r computes attention(Q_r, K_received, V_received)
  device r passes its current K_r,V_r to device r+1
  partial output is accumulated with online softmax (FlashAttention-style)
```

Total comm: every K/V slice traverses the ring once = `(N-1)/N · seqlen · hidden · bytes` per device. Total compute: same as single-GPU attention. Memory: `seqlen/N` worth of K/V cache per device.

## Ring vs Striped

**Ring Attention** (Liu et al. 2023): each device holds a contiguous chunk of tokens. Causal attention has a load-balance problem — early-token devices have less work than late-token devices (causal mask). Bubble at the end of the ring.

**Striped Attention** (Brandon et al. 2023): tokens interleave across devices. Each device holds tokens at positions `r, r+N, r+2N, ...`. The causal load-balance problem disappears because each device sees a uniform mix of early and late tokens. Standard in 2026.

## When you need CP

Long-context training. Specifics:

- Llama 4 (1M / 10M context) — CP across many devices
- Video / DiT pretraining — frame sequences become long
- RAG fine-tuning on document corpora
- Agent training with long trajectories

The trigger: when one sequence's KV cache exceeds one GPU's memory. For BF16 KV at 4K hidden dim, 1M tokens is 16 GB just for KV — needs CP=2+ on H100.

## Dynamic Context Parallelism (Megatron-Core, Jan 2026)

Static CP picks one `cp_size` for the whole training run. Variable-length workloads waste it: a packed batch with one 1M-token sequence and many 4K sequences runs all of them at `cp_size=8`, including the 4K sequences which don't need it.

Dynamic CP picks `cp_size` per microbatch (powers of 2: 1, 2, 4, 8, ...). Pre-builds all CP communicator groups at startup. Uses THD (tokens-then-headdim) packed layout. Selection is based on actual sequence length in the microbatch.

Reported: ~1.48× speedup on Llama-13B with GitHub-data sequence-length distribution; >35% end-to-end at multi-thousand-GPU scale.

Reference: [developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/).

## CP vs the other axes

CP communicates per-attention-block (every transformer layer has one). It is bandwidth-heavy at long context. Composes with TP (the same TP all-reduce per attention output stays in place; CP adds the ring around the inside).

Composition rule:
- Within an attention head: TP shards heads, CP shards sequence
- Both can run together: TP × CP within a node
- Across nodes: usually FSDP or PP

## torchtitan / Megatron-Core CP

```python
# Megatron-Core
from megatron.core.parallel_state import initialize_model_parallel
initialize_model_parallel(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    context_parallel_size=4,
)

# Then use Megatron-Core's TransformerEngine attention which is CP-aware
```

torchtitan exposes CP via DeviceMesh:

```python
mesh = init_device_mesh("cuda", (dp, tp, cp), ("dp", "tp", "cp"))
# block.attention is a CP-aware attention impl that takes mesh["cp"]
```

The attention kernel (FlashAttention varlen with CP, or NVIDIA Transformer Engine attention) does the actual ring work.

## Build steps

CP needs long sequences to be meaningful. On 2 GPUs at home:

1. Read the Ring Attention paper end-to-end ([arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)).
2. Read the Striped Attention paper ([arxiv.org/abs/2311.09431](https://arxiv.org/abs/2311.09431)).
3. Run the small CP-style ring demo in `cp_ring_demo.py` — it does the K/V passing pattern on tiny tensors so you can see the rotation.
4. If you have access to torchtitan + Llama config, run a CP=2 step at 64K context. Compare to the same model at 8K with no CP.

## Reference

- Ring Attention: [arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)
- Striped Attention: [arxiv.org/abs/2311.09431](https://arxiv.org/abs/2311.09431)
- Dynamic CP (Megatron-Core, Jan 2026): [developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
- torchtitan CP recipe: [github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py)
- Llama 4 long-context note (Meta engineering): [engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
