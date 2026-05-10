"""
Measure KL divergence between a quantized model and its BF16 reference.

The 2026 standard quantization quality metric — replaces perplexity for this purpose.

Run:
    pip install torch transformers datasets
    python measure_kl_divergence.py
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main(reference_model_id: str, quant_model_path: str, n_samples: int = 256, max_len: int = 1024):
    print(f"Reference: {reference_model_id}")
    print(f"Quantized: {quant_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(reference_model_id)

    # Calibration set — use a held-out chunk of UltraChat (NOT the same data used for
    # AWQ/GPTQ calibration; we want eval data disjoint from calibration data).
    print(f"Loading {n_samples} eval samples...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    ds = ds.shuffle(seed=123).select(range(n_samples))

    print("Loading reference (BF16)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        reference_model_id, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    ref_model.eval()

    print("Loading quantized model...")
    quant_model = AutoModelForCausalLM.from_pretrained(
        quant_model_path, device_map="cuda:0",  # same GPU is fine for small models
    )
    quant_model.eval()

    total_kl = 0.0
    n_tokens = 0

    for i, ex in enumerate(ds):
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        enc = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True).to("cuda:0")

        with torch.inference_mode():
            ref_logits = ref_model(enc.input_ids).logits.float()
            quant_logits = quant_model(enc.input_ids).logits.float()

        # KL(P_ref || P_quant) — penalizes quantized model when it shifts probability away
        # from where the reference model put it.
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_probs = ref_log_probs.exp()
        quant_log_probs = F.log_softmax(quant_logits, dim=-1)

        # Per-position KL = sum over vocab of  P_ref · (log P_ref - log P_quant)
        kl = (ref_probs * (ref_log_probs - quant_log_probs)).sum(dim=-1)

        total_kl += kl.sum().item()
        n_tokens += kl.numel()

        if (i + 1) % 25 == 0:
            running_mean = total_kl / n_tokens
            print(f"  [{i + 1}/{n_samples}]  mean KL so far: {running_mean:.4f}")

    mean_kl = total_kl / n_tokens
    print(f"\nMean per-token KL divergence: {mean_kl:.4f}")
    print(f"Total tokens evaluated:       {n_tokens}")
    print()
    print("Interpretation (rough 2026 community guidelines):")
    print("  < 0.01    Indistinguishable")
    print("  0.01-0.05 Excellent — production-ready")
    print("  0.05-0.15 Good — minor degradation")
    print("  0.15-0.50 Noticeable — okay for some uses")
    print("  > 0.50    Significant — investigate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quantized", required=True, help="Path to quantized model dir")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()
    main(args.reference, args.quantized, args.n_samples, args.max_len)
