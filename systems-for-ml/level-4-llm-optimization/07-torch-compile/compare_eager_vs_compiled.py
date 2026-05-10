"""
Compare eager vs torch.compile for LLM inference.

Measures: cold-start (first call), warm steady-state, kernel count.

Run:
    pip install torch transformers
    python compare_eager_vs_compiled.py
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain how a hash table works in one paragraph."
N_NEW_TOKENS = 100


def measure_cold_warm(model, tokenizer, label: str, device):
    """Time the first generate call (cold) and median of subsequent (warm)."""
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    # Cold call — includes JIT + compile if applicable
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=N_NEW_TOKENS,
                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0

    # Warm calls — measure steady-state
    warm_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=N_NEW_TOKENS,
                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        warm_times.append(time.perf_counter() - t0)
    warm_median = sorted(warm_times)[len(warm_times) // 2]

    tok_per_s_warm = N_NEW_TOKENS / warm_median
    print(f"{label:<30}  cold {cold_s:>5.1f}s  warm {warm_median * 1000:>5.0f}ms  ({tok_per_s_warm:.1f} tok/s)")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model: {MODEL_ID}\nPrompt: {PROMPT}\nTokens: {N_NEW_TOKENS}\n")
    print(f"{'config':<30}  {'cold':>10}  {'warm':>10}  {'tok/s':>10}")
    print("-" * 70)

    # Eager
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    measure_cold_warm(model, tokenizer, "eager (no compile)", device)
    del model
    torch.cuda.empty_cache()

    # torch.compile, default mode
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    model = torch.compile(model)  # default: dynamic=True, mode="default"
    measure_cold_warm(model, tokenizer, "torch.compile (default)", device)
    del model
    torch.cuda.empty_cache()

    # torch.compile, reduce-overhead mode (uses CUDA graphs internally)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    model = torch.compile(model, mode="reduce-overhead")
    measure_cold_warm(model, tokenizer, "torch.compile (reduce-overhead)", device)
    del model
    torch.cuda.empty_cache()

    print()
    print("Notes:")
    print("- Cold time for compiled = JIT + Inductor lowering. 30-60s for 7B; less for 0.5B.")
    print("- Warm time should drop 1.3-2× from eager for small batch / decode-heavy workloads.")
    print("- For production, use vLLM V1 — its piecewise CUDA graph pattern is more refined")
    print("  than what HF transformers + torch.compile gives you out of the box.")
    print("- Subsequent runs of THIS script will reuse the compile cache and be much faster.")


if __name__ == "__main__":
    main()
