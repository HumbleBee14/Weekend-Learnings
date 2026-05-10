# 09 — MoE on Mac

## The math that changed local

Decoding a transformer at batch=1 is **memory-bandwidth-bound**, not compute-bound. Tokens-per-second is roughly `bytes_per_token_read / DRAM_bandwidth`.

For a dense 70B at fp16: ~140 GB to read per token. Even at M5 Max's 614 GB/s, that ceilings around 4 tok/s. At 4-bit (~35 GB / token): around 17 tok/s. Painful.

For a sparse Mixture-of-Experts where only a fraction of experts route per token, you only stream the active experts plus the always-on attention/embedding/router weights.

```
  Llama 4 Scout
  Total params:    109B   (4-bit ~ 55 GB on disk and RAM)
  Active per tok:  17B    (4-bit ~  9 GB streamed)
  Result on M5 Max:  ~50+ tok/s
```

Translation: bigger total model, faster decode. Counterintuitive on cloud, decisive on Mac.

## Models that matter on Mac in 2026

| Model | Active | Total | 4-bit RAM | Notes |
|---|---|---|---|---|
| Llama 4 Scout | 17B | 109B | ~55 GB | First MoE flagship that fits on a 64-GB Mac |
| Llama 4 Maverick | 17B | 400B | ~200 GB | Distributed only (Topic 15) |
| Qwen3-Next 80B-A3B | 3B | 80B | ~40 GB | Tiny per-tok bandwidth — fast on M3 Max |
| Qwen3-Next 80B-A22B | 22B | 80B | ~40 GB | Heavier active pull, higher quality |
| DeepSeek V3.2 (small) | varies | varies | varies | 256-expert routing, MTP heads |
| Mixtral 8x22B | 39B | 141B | ~70 GB | Older but still solid baseline |

(Numbers are approximate; precise active-bandwidth depends on top-k routing.)

## Why the active-params number is what you read

```
       per-token bandwidth =
           always-on (embed + attn + router + LN + lm_head)
         + sum of activated experts at this token

  dense:   always-on + ALL FFN weights
  MoE:     always-on + top-k of N expert weights
```

`top_k` is usually 1 or 2 in 2026 (Mixtral was 2; Llama 4 is 1; DeepSeek varies). Memory-bound decode scales with bytes read, so MoE moves the throughput floor up a lot.

The catch: prefill (long prompts) is compute-bound and uses *all* experts when batched, because different tokens hit different experts. Long-context prefill on MoE is not 4× faster than dense — it is roughly the same. The win is decode.

## Routing — minimum viable picture

Each MoE layer:

```
                                  +----- expert_0 -----+
       hidden state  -----+       |                    |
                          v       +----- expert_1 -----+
                 +-----------+    |       ...
                 |  Router   | -- + ------ ...
                 |  (linear) |    |
                 +-----------+    +----- expert_N -----+
                          |
                          v
                  top-k softmax routes -> gather chosen experts
                          |
                          v
                weighted sum of expert outputs
```

The router is a small linear layer. Its outputs are scored, top-k experts are selected, and only those experts run for that token. The **gather** is what makes MoE awkward on hardware: tokens in a batch route to different experts, so naive implementation kills throughput.

MLX, vLLM-MLX, and llama.cpp now have proper grouped-matmul kernels for MoE expert dispatch (Apple ML's "M5 + MoE" notes describe this for MLX). Without those, you get the active-param size on disk but not the active-param speed.

## Loading a Llama-4-Scout on Mac

```bash
pip install mlx-lm
python -m mlx_lm.generate \
    --model mlx-community/Llama-4-Scout-17B-A109B-Instruct-4bit \
    --prompt "Explain why MoE is faster on Mac than dense at the same quality." \
    --max-tokens 200
```

On an M5 Max 128 GB this should sustain 50+ tok/s. On M3 Max 64 GB you can run it but with thinner headroom — close other apps; 4-bit KV (Topic 10) helps.

For the smaller-active alternative:

```bash
python -m mlx_lm.generate \
    --model mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
    --prompt "..." \
    --max-tokens 200
```

A3B (3B active) is the absolute speed king on Mac — the active stream per token is tiny.

## Comparing dense and MoE empirically

What to measure:

1. Decode tok/s (bandwidth-limited).
2. Prefill tok/s (compute-limited).
3. Peak RAM during a 4k-prompt + 1k-decode run.
4. Quality on a small benchmark slice (MMLU-Pro 200 questions).

Expected pattern:

- Decode tok/s: MoE >> dense (sometimes 2–3×).
- Prefill tok/s: MoE ~ dense, sometimes worse on small batches.
- Peak RAM: MoE > dense at matched quality (you pay for storing all experts).
- Quality: MoE ~ dense at matched-active flops, often better at matched-decode-speed.

## Common pitfalls

1. **Confusing total size with bandwidth.** Llama 4 Scout is "109B" on disk but the per-token streaming cost is closer to 17B. Buy RAM for total; predict speed from active.
2. **Old MoE kernels.** llama.cpp's early MoE path serialized expert calls. Update to a recent build (FP4 + MoE kernels, May 2026).
3. **Long-prompt MoE myth.** MoE doesn't speed up prefill — it speeds up decode. Long batched prompts are still compute-bound.
4. **Ignoring router top-k.** A model that says it's "17B active" with `top_k=1` reads less than one with `top_k=2` — different per-token cost.
5. **Skipping KV cache quantization.** MoE saves on weights, not KV. A 100k-token MoE run still needs 4-bit KV (Topic 10) to fit.

## References

- Apple ML — Exploring LLMs with MLX on M5: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Llama 4 model card: https://ai.meta.com/research/llama-4/
- Qwen3-Next: https://qwenlm.github.io/blog/qwen3-next/
- DeepSeek V3 paper: https://arxiv.org/abs/2412.19437
- mlx-community Llama 4: https://huggingface.co/mlx-community/Llama-4-Scout-17B-A109B-Instruct-4bit
- llama.cpp MoE PRs (May 2026): https://github.com/ggerganov/llama.cpp/pulls?q=is%3Apr+moe
- Mixtral paper (origins): https://arxiv.org/abs/2401.04088
