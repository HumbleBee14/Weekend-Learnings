"""Trace vLLM v1's attention dispatch. Lab-notebook style.

Prereq:
    pip install vllm   # 0.7+
    export VLLM_USE_V1=1

Run:
    python vllm_dispatch_trace.py

What you want to see:
    - Prefill goes through flashinfer.BatchPrefillWithPagedKVCacheWrapper (on Ampere/Hopper).
    - Decode goes through flashinfer.BatchDecodeWithPagedKVCacheWrapper.
    - On Blackwell (SM100), prefill may go via FA4 wrapped behind FlashInfer's dispatch.
    - The default fallback when FlashInfer is unavailable is vLLM's native Triton paged attention.
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_USE_V1", "1")


def main() -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise SystemExit("pip install vllm")

    # Tiny model for the trace; the dispatch chain is the same as a large model.
    llm = LLM(model="facebook/opt-125m", enforce_eager=False, gpu_memory_utilization=0.5)
    prompts = [
        "Hello",
        "The capital of France is",
        "Once upon a time " + " ".join(["very"] * 200),
        "x " * 1000,
    ]
    params = SamplingParams(max_tokens=8)
    out = llm.generate(prompts, params)
    for o in out:
        print(f"[{len(o.prompt_token_ids)} prompt toks] -> {o.outputs[0].text!r}")

    # Inspect which backend got selected.
    # vLLM v1 logs the chosen backend at startup. Grep the stdout for lines like:
    #   "Using attention backend: flashinfer"
    # or  "Using attention backend: flash_attn"
    # or  "Using attention backend: triton_attn"
    print("\nCheck the vLLM startup log for 'Using attention backend: ...' line.")
    print("If FlashInfer is selected, you've confirmed the dispatch.")


if __name__ == "__main__":
    main()
