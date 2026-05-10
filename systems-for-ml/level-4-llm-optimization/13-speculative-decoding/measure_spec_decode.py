"""
Measure speculative decoding speedup using vLLM's offline API.

Compares:
  - Baseline (no spec decode)
  - n-gram spec decode (no draft head needed)
  - EAGLE-3 (if you have a model with a pre-trained EAGLE head)

Run:
    pip install vllm
    python measure_spec_decode.py
"""

import time

from vllm import LLM, SamplingParams


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # adjust to a model that fits your GPU
N_NEW_TOKENS = 256

# Code-generation prompt — n-gram does well on repetitive structure
PROMPTS = [
    "Write a Python function `quicksort(arr)` that sorts a list in place. Include docstring and one example.",
    "Implement a `Stack` class in Python with push, pop, peek, is_empty, and __len__. Include type hints.",
    "Write a Python function `is_prime(n: int) -> bool` using a sieve approach. Include comments.",
    "Implement binary search on a sorted list in Python. Return -1 if not found. Include a small test.",
]


def run_once(llm: LLM, label: str):
    sp = SamplingParams(temperature=0.0, max_tokens=N_NEW_TOKENS)

    # Warmup
    llm.generate([PROMPTS[0]], sp)

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, sp)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    print(f"{label:<30}  {elapsed:>6.2f}s  {total_tokens:>5} tokens  {throughput:>6.1f} tok/s")

    # Per-prompt details if metrics available
    return outputs


def main():
    print(f"Model: {MODEL_ID}, N_NEW_TOKENS: {N_NEW_TOKENS}, n_prompts: {len(PROMPTS)}\n")

    # ---- Baseline ----
    print(f"{'config':<30}  {'time':>8}  {'tokens':>8}  {'throughput':>12}")
    print("-" * 75)

    llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.85)
    run_once(llm, "baseline")
    del llm

    # ---- n-gram spec decode ----
    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.85,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
        },
    )
    run_once(llm, "n-gram (k=5)")
    del llm

    # ---- EAGLE-3 ----
    # Requires a model with a pre-trained EAGLE head. Skip if not available.
    # Llama-3.1-8B-Instruct has community EAGLE heads.
    # See https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B
    print()
    print("EAGLE-3 / P-EAGLE require a model with a pre-trained spec head.")
    print("See vLLM docs: https://docs.vllm.ai/en/latest/features/spec_decode.html")

    print()
    print("What to look for:")
    print("- n-gram should give ~1.5-2× on code-generation workloads (lots of repetition)")
    print("- EAGLE-3 typically 2-2.5× across chat/code/general workloads")
    print("- P-EAGLE (vLLM v0.16+) up to another 1.69× over EAGLE-3")
    print("- Always verify quality (Topic 06's lm-eval-harness)")


if __name__ == "__main__":
    main()
