"""
Baseline quantization comparison: BF16, FP16, INT8 weight-only on the same model.

Measures memory footprint and decode throughput. The starting point for the
quantization quality/cost frontier table you'll build across Topics 01-05.

Run:
    pip install torch transformers bitsandbytes accelerate
    python baseline_measurements.py
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # small for fast iteration
PROMPT = "Explain how a hash table works in one paragraph."
N_NEW_TOKENS = 100
N_RUNS = 5


def measure(model, tokenizer, label: str, device):
    """Generate, time it, return (tokens/sec, ms_per_token)."""
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    # Warmup
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=N_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median_s = sorted(times)[len(times) // 2]
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tok_per_s = new_tokens / median_s
    ms_per_tok = median_s * 1000 / new_tokens

    # Peak memory used during inference
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()

    print(f"{label:<25}  {tok_per_s:>7.1f} tok/s  {ms_per_tok:>5.1f} ms/tok  peak {peak_mb:>7.1f} MB")
    return tok_per_s, ms_per_tok, peak_mb


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {PROMPT[:60]}...")
    print(f"max_new_tokens={N_NEW_TOKENS}, runs={N_RUNS}, reporting median.\n")
    print(f"{'config':<25}  {'tok/s':>9}  {'ms/tok':>10}  {'peak mem':>10}")
    print("-" * 70)

    # ---- BF16 baseline ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    measure(model, tokenizer, "BF16 (baseline)", device)
    del model
    torch.cuda.empty_cache()

    # ---- FP16 ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()
    measure(model, tokenizer, "FP16", device)
    del model
    torch.cuda.empty_cache()

    # ---- INT8 weight-only (bitsandbytes LLM.int8) ----
    # W8A16 — weights in 8-bit, activations stay FP16.
    int8_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=int8_config, device_map="auto",
    )
    model.eval()
    measure(model, tokenizer, "INT8 (W8A16, bnb)", device)
    del model
    torch.cuda.empty_cache()

    # ---- NF4 (4-bit weight-only, bitsandbytes) ----
    # W4A16 — weights in 4-bit NormalFloat, activations stay FP16.
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=nf4_config, device_map="auto",
    )
    model.eval()
    measure(model, tokenizer, "NF4 (W4A16, bnb)", device)
    del model
    torch.cuda.empty_cache()

    print("\nNotes:")
    print("- BF16/FP16 are the baselines. INT8 weight-only halves memory at small speed cost.")
    print("- NF4 quarters memory but slower than INT8 in bnb (kernel quality varies).")
    print("- Topic 02 covers FP8/NVFP4 which need Hopper+/Blackwell hardware.")
    print("- Topic 03 covers proper PTQ (AWQ/GPTQ) with calibration via llm-compressor.")
    print("- Topic 06 covers actually MEASURING quality (lm-eval-harness, KL-divergence) —")
    print("  these throughput numbers are meaningless without it.")


if __name__ == "__main__":
    main()
