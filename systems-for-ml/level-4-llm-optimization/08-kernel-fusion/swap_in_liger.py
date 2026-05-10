"""
Swap PyTorch's RMSNorm and SwiGLU for Liger-Kernel's fused versions.
Measure the delta.

Run:
    pip install torch transformers liger-kernel
    python swap_in_liger.py
"""

import time
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain how a CPU cache hierarchy works in two paragraphs."
N_NEW_TOKENS = 128


def measure(model, tokenizer, label: str, device, n_runs: int = 5):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    # Warmup
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=N_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median = sorted(times)[len(times) // 2]
    tok_per_s = N_NEW_TOKENS / median
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"{label:<40}  {median * 1000:>6.0f}ms  {tok_per_s:>6.1f} tok/s  peak {peak_mb:>6.0f} MB")
    torch.cuda.reset_peak_memory_stats()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model: {MODEL_ID}\nPrompt: {PROMPT}\nN_NEW_TOKENS: {N_NEW_TOKENS}\n")
    print(f"{'config':<40}  {'time':>8}  {'tok/s':>8}  {'peak mem':>10}")
    print("-" * 80)

    # Baseline — PyTorch's stock layers
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    measure(model, tokenizer, "stock PyTorch layers", device)
    del model
    torch.cuda.empty_cache()

    # With Liger-Kernel applied
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
        apply_liger_kernel_to_qwen2(
            rms_norm=True,
            rope=True,
            swiglu=True,
            cross_entropy=False,  # not used in inference
            model=model,
        )
        model.eval()
        measure(model, tokenizer, "Liger-Kernel (RMSNorm + RoPE + SwiGLU)", device)
        del model
        torch.cuda.empty_cache()
    except ImportError:
        print("Liger-Kernel not installed — skipping. pip install liger-kernel")

    # With torch.compile (for comparison)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    model = torch.compile(model, mode="reduce-overhead")
    measure(model, tokenizer, "torch.compile (reduce-overhead)", device)
    del model
    torch.cuda.empty_cache()

    print()
    print("Notes:")
    print("- Liger-Kernel's win is biggest at low batch / decode-heavy workloads.")
    print("- For prefill-heavy or large-batch workloads, the win is smaller (compute-bound regime).")
    print("- Liger and torch.compile compose: apply Liger first, then torch.compile the result.")


if __name__ == "__main__":
    main()
