# 10 — KV Cache Quantization + Speculative Decoding

Two independent levers, both essential on Mac. Together they make 100k-context inference and 2–3× decode speedups practical on consumer hardware.

## Part 1 — KV cache quantization

### Why it matters more on Mac than on a server

A single 32k-context Llama-3-8B run keeps a KV cache around 4–6 GB at fp16. Push to 128k and the cache dominates RAM. On a 64 GB Mac, KV is what kicks you into swap before model weights do.

```
  weights (constant)         |--------|         (~5 GB at 4-bit)
  KV at  fp16, 32k context   |--------------------------------|   (~5 GB)
  KV at  4-bit, 32k context  |--------|                           (~1.3 GB)
  KV at  fp16, 128k context  ===========================...===   (~20 GB, ouch)
  KV at  4-bit, 128k context |================|                   (~5 GB, fine)
```

### What it is

Same idea as weight quantization, applied to the K and V tensors stored in the cache. Per-head per-channel scales, group size 64. 4-bit symmetric is the workhorse; 8-bit asymmetric is the safe choice when quality matters more than memory.

### Quality cost

For modern models with grouped-query attention, 4-bit KV is nearly free on perplexity (< 0.1 ppl drift on most tasks). Long-context recall at 64k+ is where you sometimes notice — needle-in-a-haystack tests can drop 1–2 percentage points. 8-bit is indistinguishable.

### Enabling it

MLX:

```bash
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --prompt "$(cat long_doc.txt)" \
    --max-tokens 1024 \
    --kv-bits 4 --kv-group-size 64
```

llama.cpp:

```bash
./llama-cli -m model.gguf --prompt "..." \
    --cache-type-k q4_0 --cache-type-v q4_0
```

vLLM-MLX:

```python
LLM(model="...", kv_cache_dtype="fp8_e5m2")  # or "int8"
```

### What to measure

1. Peak RAM at 32k / 64k / 128k context, fp16 vs 4-bit KV.
2. TTFT and decode tok/s — KV quant is mostly free on speed (it can even help by reducing bandwidth pressure during attention).
3. A small recall benchmark — needle-in-a-haystack, multi-doc QA — to verify quality.

## Part 2 — Speculative decoding on Mac

### The idea (recap from Level 4 Topic 13)

Run a small fast draft model. It proposes K tokens. Verify them with the target model in **one** parallel forward pass. Accept the longest correct prefix. Net throughput up if acceptance rate is high enough that one target step replaces 2–3 sequential decode steps.

```
  draft  : t1 t2 t3 t4 t5  (cheap, sequential)
            |  |  |  |  |
            v  v  v  v  v
  target : forward(t0..t4) -> verify all 5 in parallel
            |  |  |  |  |
            +--+--+--x  x   (accept first 3)
            => committed: t1 t2 t3, fall back to target sample for next
```

Speedup ~ acceptance_rate × draft_length / (1 + draft_compute_ratio).

### What's different on Apple Silicon in 2026

Three options, ranked by readiness on MLX:

1. **Apple QuantSpec** — *self-speculative*. The same model, quantized harder for the draft path, sharing the verifier's KV cache via a hierarchical 4-bit cache. No second model to load. Documented in Apple ML's QuantSpec post; the technique informs MLX's spec-decode direction though the exact CLI surface in `mlx_lm` evolves.
2. **EAGLE-3** — uses a tiny EAGLE head trained on the target. Where draft weights exist for the target, this is the highest-acceptance option. MLX support landed in late 2025; pick a model that has both base and EAGLE-3 head published on `mlx-community`.
3. **n-gram / lookup decoding** — no model at all; the draft is "tokens that already appeared in the prompt." Free, often gets 1.3–1.6× on code completion where repetition is high.

```
  +----------------+    +----------------+    +-----------------+
  | n-gram draft   |    |  QuantSpec     |    |  EAGLE-3 head   |
  | zero cost      | -> |  shared model  | -> |  trained head   |
  | 1.3-1.6x       |    |  2-3x typical  |    |  2.5-3.5x       |
  +----------------+    +----------------+    +-----------------+
```

### Enabling QuantSpec

Recent `mlx-lm`:

```bash
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --prompt "..." --max-tokens 256 \
    --speculative quantspec \
    --kv-bits 4
```

### Enabling EAGLE-3

```bash
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --draft-model mlx-community/Qwen2.5-7B-EAGLE3 \
    --num-draft-tokens 5
```

The acceptance rate is the metric to watch. < 50% means the draft is fighting the target — disable it. > 70% on coding workloads is realistic for EAGLE-3.

### What stacks with what

- 4-bit KV + spec decode stack cleanly: spec decode does parallel verification, which is bandwidth-bound on KV; smaller KV makes verification faster too.
- 4-bit KV + MoE (Topic 09) stack — long-context Llama-4-Scout on 64 GB needs 4-bit KV.
- Spec decode + continuous batching (vLLM-MLX) is more nuanced: at high concurrency, the win shrinks because the GPU is already saturated. Best at batch ≤ 2.

## Common pitfalls

1. **Pretending KV is free.** It dominates memory at long context. If you don't quantize it, you swap before you finish prefill.
2. **Spec decode at high concurrency.** vLLM with 32 concurrent requests is GPU-saturated; spec decode adds overhead with little gain. Disable it for high-throughput regimes.
3. **Mixing draft and target tokenizers.** EAGLE-3 head must be trained on the same tokenizer as target. Off-the-shelf `mlx-community` checkpoints get this right; rolling your own does not by default.
4. **Reading "X% speedup" without checking acceptance rate.** A 2× wallclock speedup at 80% acceptance is real. A 1.05× at 35% acceptance is variance.

## References

- Apple QuantSpec: https://machinelearning.apple.com/research/quantspec
- EAGLE-3 paper: https://arxiv.org/abs/2503.01840
- mlx-lm KV cache quant: https://github.com/ml-explore/mlx-lm
- llama.cpp KV cache quant: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
- Original speculative decoding: https://arxiv.org/abs/2211.17192
- vLLM spec decode: https://docs.vllm.ai/en/latest/features/spec_decode.html
