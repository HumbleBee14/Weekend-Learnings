# 09 — MoE on Mac

## Files

- `CONCEPTS.md` — the active-params bandwidth math, models that matter (Llama 4 Scout, Qwen3-Next, DeepSeek V3.2), routing picture, dense-vs-MoE empirical comparison plan.
- `moe_vs_dense.py` — runs a dense 7B and a MoE checkpoint through `mlx_lm`, reports decode tok/s, prefill tok/s, and peak RAM.

## Quickstart

```bash
pip install mlx-lm psutil
python moe_vs_dense.py \
    --dense  mlx-community/Qwen2.5-7B-Instruct-4bit \
    --moe    mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
    --prompt "Explain MoE active-parameter routing in 200 words." \
    --max-tokens 256
```

## Expected output

```
=== Dense  mlx-community/Qwen2.5-7B-Instruct-4bit ===
prefill tok/s: ~1800
 decode tok/s: ~225
   peak RAM:    5.2 GB

=== MoE    mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit ===
prefill tok/s: ~1500
 decode tok/s: ~310
   peak RAM:   42.0 GB

Decode delta:  +38% (MoE faster at higher quality and 8x total params)
```

Numbers vary widely by hardware. The qualitative pattern — MoE wins decode tok/s while paying total-RAM cost — should hold on any M-series with enough RAM.

## Try

- Swap the MoE for `mlx-community/Llama-4-Scout-17B-A109B-Instruct-4bit` (needs ~64 GB free). Same script.
- Increase `--prompt-tokens 8000` (a long prompt) and re-measure prefill — MoE's prefill advantage shrinks (compute-bound regime).
- Combine with Topic 10's `--kv-bits 4` to keep memory in check on long-context MoE runs.
- Add a small MMLU subset and confirm MoE delivers higher quality at higher tok/s — both axes win simultaneously, which is the real story.

## Where this goes

Topic 10 keeps long-context MoE alive on a 64 GB Mac via 4-bit KV. Topic 11 is where this MoE shows up as the chat brain in the agentic loop.
