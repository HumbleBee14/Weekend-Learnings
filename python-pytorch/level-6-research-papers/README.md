# Level 6 — Research Paper Implementations

Implement the key architectural innovations that power modern LLMs from scratch.
Goal: understand the algorithm deeply enough to build on top of it.

## Philosophy

Each notebook has two parts:
1. **Conceptual PyTorch** — clean, readable implementation that matches the paper math (CPU ok)
2. **Optimized version** — production-grade with CUDA/Triton where needed (Colab GPU)

## Papers / Topics

| # | Notebook | Paper | Links | Why it matters |
|---|----------|-------|-------|----------------|
| 01 | flash-attention | FlashAttention — Dao et al. 2022 | [arxiv](https://arxiv.org/abs/2205.14135) · [FA-2](https://arxiv.org/abs/2307.08691) · [github](https://github.com/Dao-AILab/flash-attention) | Standard attention OOM on long sequences. Flash Attention fixes this — used in every modern LLM |
| 02 | rope-embeddings | RoFormer — Su et al. 2021 | [arxiv](https://arxiv.org/abs/2104.09864) · [blog](https://blog.eleutherai.org/rotary-embeddings) | How LLaMA/Mistral handle position info. Enables longer context than learned embeddings |
| 03 | grouped-query-attention | GQA — Ainslie et al. 2023 | [arxiv](https://arxiv.org/abs/2305.13245) | GQA reduces KV cache size — used in LLaMA-2, Mistral, Gemma |
| 04 | kv-cache | Systems technique | [vLLM blog](https://blog.vllm.ai/2023/06/20/vllm.html) · [HF docs](https://huggingface.co/docs/transformers/kv_cache) | How LLMs do fast inference. Without this, every token regenerates the whole sequence |
| 05 | mixture-of-experts | Switch Transformers — Fedus et al. 2022 | [arxiv](https://arxiv.org/abs/2101.03961) · [Mixtral](https://arxiv.org/abs/2401.04088) · [HF blog](https://huggingface.co/blog/moe) | How Mixtral/GPT-4 scale params without scaling compute — sparse activation |
| 06 | mamba-ssm | Mamba — Gu & Dao 2023 | [arxiv](https://arxiv.org/abs/2312.00752) · [github](https://github.com/state-spaces/mamba) · [annotated](https://srush.github.io/annotated-mamba/hard.html) | State Space Models — linear-time alternative to attention, strong on long sequences |
| 07 | turboquant | TurboQuant — Google ICLR 2026 | [arxiv](https://arxiv.org/abs/2503.02520) · [github](https://github.com/tonbistudio/turboquant-pytorch) | KV cache quantization — 6x memory reduction, 8x decode speedup on H100 |

## Prerequisites

Complete Levels 1–5 first, especially:
- `level-4/04-transformer.ipynb` — you need to understand standard attention cold
- `level-5/04-lora-qlora.ipynb` — parameter efficiency concepts apply here too

## Colab vs Local

```
CPU (local or Colab):      01-07 conceptual implementations — understand the algorithm
GPU T4 (Colab free):       flash attention real benchmarks, MoE training
GPU A100 (Colab Pro):      Mamba full sequence experiments, long-context flash attention, TurboQuant benchmarks
```
