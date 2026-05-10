# 10 — KV Cache Quantization + Speculative Decoding

## Files

- `CONCEPTS.md` — why KV dominates memory at long context, 4-bit KV mechanics, QuantSpec / EAGLE-3 / n-gram speculative decoding on MLX, what stacks with what.
- `kv_quant_bench.py` — runs the same 32k-token prompt with fp16 KV and 4-bit KV on `mlx-lm`, reports peak RAM, TTFT, decode tok/s.

## Quickstart

```bash
pip install mlx-lm
python kv_quant_bench.py \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --context-tokens 32000 \
    --max-tokens 128
```

## Expected output

```
=== fp16 KV ===
  prefill tok/s: ~1700
  decode tok/s:  ~210
  KV bytes:      ~4.6 GB

=== 4-bit KV ===
  prefill tok/s: ~1700
  decode tok/s:  ~225
  KV bytes:      ~1.3 GB

KV memory delta: 71% reduction.
```

Numbers vary by hardware. The pattern — same speed, ~70% smaller KV — is the point.

## Try

- Push `--context-tokens 100000`. With fp16 KV on a 64 GB Mac the run will swap and tok/s collapses; with 4-bit KV it just works.
- Try a QuantSpec-style self-speculative path (check `mlx_lm`'s current spec-decode flags — the surface evolves) and re-measure. Watch acceptance rate. On a coding prompt expect 60–80% and a ~2× decode wallclock speedup.
- Add an EAGLE-3 head: `--draft-model mlx-community/Qwen2.5-7B-EAGLE3 --num-draft-tokens 5`.
- For llama.cpp comparison: same model in GGUF with `--cache-type-k q4_0 --cache-type-v q4_0`. Quality should match, speed will be slower than MLX (Topic 03).

## Where this goes

Topic 11 builds the agentic loop on top of these primitives — sub-100ms TTFT for autocomplete depends on small-prompt decode, which spec decode accelerates. Topic 12 fine-tunes; Topic 13 does preference learning on top of that. The KV quant path is what keeps 100k-context evaluations fittable on a developer's laptop.
