# 06 — Quality Evaluation

## Why this topic exists

Topics 01-05 produced a quantized model. The throughput numbers look great. **They are meaningless without a quality measurement.**

"Feels fine" testing fails predictably:

- A handful of test prompts pass — quantized model breaks on long-context, rare tokens, multi-step reasoning.
- The chat output looks coherent — math accuracy dropped 15 points.
- MMLU score dropped 3 points — but the model is now unusable for code generation.

Two measurements that *do* work:

1. **KL divergence vs the BF16 reference** — distributional, sensitive to quantization noise
2. **Task-suite eval** with `lm-evaluation-harness` — MMLU, GSM8K, HumanEval, etc.

Use both. They catch different failure modes.

## The 2026 methodology shift — KL divergence over perplexity

Perplexity has a well-known flaw: token-level errors cancel. A quantized model can have:

- Higher perplexity → "quality dropped"
- Same perplexity, different distribution shape → quality dropped, perplexity didn't catch it

KL divergence compares the *full output distribution* of the quantized model vs the BF16 reference at every position:

```
KL(P_ref || P_quant) = sum over vocab of  P_ref(token) · log(P_ref(token) / P_quant(token))
```

For a calibration set of prompts, average per-token KL across all positions. Lower = quantized model's distribution is closer to reference.

Why KL is better:

- Catches "the model still picks the right top-1 token but the second-best probability is now wildly different" (perplexity misses this; downstream behavior changes)
- Catches subtle redistributions that "feels fine" can't see
- Standardized across models — KL of 0.05 is comparable across base models

In 2026, **KL divergence has overtaken perplexity** as the standard quantization quality metric. The local-LLM community popularized it; production teams adopted it.

## How to compute KL divergence

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                  torch_dtype=torch.bfloat16, device_map="cuda:0")
quant_model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-1.5B-Instruct-FP8",
                                                    device_map="cuda:1")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

calib_texts = load_calibration_set(n=512)  # representative prompts

total_kl = 0.0
n_tokens = 0
for text in calib_texts:
    ids = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    
    with torch.inference_mode():
        ref_logits = ref_model(ids.input_ids.to("cuda:0")).logits
        quant_logits = quant_model(ids.input_ids.to("cuda:1")).logits.to("cuda:0")
    
    ref_probs = F.softmax(ref_logits, dim=-1)
    quant_log_probs = F.log_softmax(quant_logits, dim=-1)
    
    # KL(ref || quant) per position
    kl = (ref_probs * (ref_probs.log() - quant_log_probs)).sum(dim=-1)
    
    total_kl += kl.sum().item()
    n_tokens += kl.numel()

mean_kl = total_kl / n_tokens
print(f"Mean per-token KL: {mean_kl:.4f}")
```

Target ranges from community measurements:

```
KL value       Interpretation
─────────────────────────────────────────
< 0.01         Indistinguishable from reference
0.01-0.05      Excellent — production-ready
0.05-0.15      Good — minor degradation, usually fine
0.15-0.50      Noticeable — okay for some uses
> 0.50         Significant — model behavior diverges meaningfully
```

(These are rough; calibrate against your own model's known good/bad recipes.)

## Task suite — `lm-evaluation-harness`

The other half of quality measurement. Run a suite of standardized benchmarks:

- **MMLU** (5-shot) — broad knowledge across 57 subjects
- **GSM8K** — grade-school math, tests reasoning
- **HumanEval** — Python code generation
- **HellaSwag** — common-sense reasoning
- **ARC-Easy / ARC-Challenge** — science Q&A

The 2026 standard practice: run each benchmark before and after quantization. Tolerance: **<1% drop on MMLU**, similar tolerances on others.

```bash
pip install lm-eval

# Reference (BF16)
lm-eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16 \
    --tasks mmlu,gsm8k,arc_easy,hellaswag \
    --batch_size 8 \
    --output_path results/bf16

# Quantized (FP8)
lm-eval --model vllm \
    --model_args pretrained=./Qwen2.5-1.5B-Instruct-FP8 \
    --tasks mmlu,gsm8k,arc_easy,hellaswag \
    --batch_size 8 \
    --output_path results/fp8
```

vLLM-loaded eval is faster than HF for big models; identical results.

## Subset evals for fast iteration

Running full MMLU takes 30+ minutes per recipe. For iteration, use subsets:

```bash
# MMLU-Pro subset (faster, harder)
lm-eval --tasks mmlu_pro_subset --limit 500 ...

# Custom subset of 100 GSM8K
lm-eval --tasks gsm8k --limit 100 ...
```

Pattern: subset evals during development, full eval before release.

## Per-layer sensitivity analysis

For diagnosing *where* a quantization recipe is hurting quality:

```
For each layer l in the model:
  Run inference on calibration set with ONLY that layer quantized
  Compute KL divergence
  Layers with high KL are sensitive — exclude them or use higher precision
```

This is how mixed-precision recipes (Unsloth Dynamic, llm-compressor's multi-compressor) decide which layers get more bits.

## What to measure for each quantization recipe

```
Recipe        Throughput     Memory     KL          MMLU drop     GSM8K drop
─────────────────────────────────────────────────────────────────────────────
BF16          baseline       baseline    0.000       baseline       baseline
FP8           1.7×           50%         0.012       -0.3 pp        -0.5 pp
NVFP4         3.5×           25%         0.078       -1.8 pp        -2.4 pp
AWQ-W4A16     1.6×           28%         0.048       -1.2 pp        -1.6 pp
IQ4_XS        N/A (CPU)      28%         0.063       -1.5 pp        -2.1 pp
IQ2_M         N/A (CPU)      14%         0.42        -8.5 pp        -15 pp
```

(Numbers illustrative; vary by model.)

## Pitfalls

1. **Using the calibration set as the eval set.** Inflates measured quality.
2. **Trusting MMLU alone.** Some quants degrade GSM8K or HumanEval much more than MMLU. Run multiple tasks.
3. **Comparing models trained differently.** A quantized Qwen 0.5B vs unquantized Llama 1B is apples-to-oranges. Always quantize FROM the same reference.
4. **KL divergence with too few samples.** ~512 prompts × 2048 tokens is the rough floor for stable KL estimates.
5. **Measuring on synthetic data.** "Lorem ipsum" KL is meaningless. Use real prompts from your domain.
6. **Stopping at "score X is acceptable."** Run a sample of outputs by hand. Scores can be acceptable while specific failure modes are not.

## What you should walk away with

For each quantization recipe in your table (Topics 01-05), measure:

- KL divergence against BF16
- MMLU drop (5-shot)
- GSM8K drop
- HumanEval drop (if you serve code)

Update the quality column in the comparison table. Now your throughput numbers mean something.

## References

- lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness
- Measuring quantization quality with KL divergence (smcleod, April 2026) — https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/
- MMLU paper — https://arxiv.org/abs/2009.03300
- GSM8K paper — https://arxiv.org/abs/2110.14168
- llm-compressor's eval examples — https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w4a4_fp4
