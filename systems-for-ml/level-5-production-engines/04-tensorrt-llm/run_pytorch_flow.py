"""
04 - TRT-LLM PyTorch flow: serve a model in-process and time generation.

Prereqs (Linux + NVIDIA + CUDA 12+, Hopper for FP8):
    pip install tensorrt-llm

Run:
    python run_pytorch_flow.py

This script uses the modern Python-first API (the post-1.0 default).
It builds the engine on import — first run is slow (5-30 min).
Subsequent runs reuse the cached engine.

For a server, use the CLI instead:
    trtllm-serve Qwen/Qwen2.5-7B-Instruct --port 8002 \
        --tp_size 1 --kv_cache_free_gpu_memory_fraction 0.9
Then hit it with the same OpenAI-client harness from Topic 01.
"""

from __future__ import annotations

import time

# tensorrt_llm imports CUDA at top level — guard so the file is at least
# readable on machines that don't have CUDA.
try:
    from tensorrt_llm import LLM, SamplingParams  # type: ignore
except ImportError:
    print("tensorrt_llm not importable — run on a CUDA box with `pip install tensorrt-llm`.")
    raise SystemExit(0)


PROMPTS = [
    "Explain paged KV cache to a database engineer in two sentences.",
    "Why is FP8 a good fit for Hopper specifically?",
    "When does TRT-LLM lose to vLLM?",
    "Sketch the difference between in-flight batching and static batching.",
]


def main() -> None:
    print("Building engine (first run is slow — 5-30 min for 7B FP8)...")
    t0 = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        dtype="bfloat16",
        # On Hopper+, enable FP8 — biggest single throughput lever.
        # quant_config={"quant_algo": "FP8"},
    )
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    sampling = SamplingParams(max_tokens=128, temperature=0.7, top_p=0.9)

    t1 = time.perf_counter()
    outputs = llm.generate(prompts=PROMPTS, sampling_params=sampling)
    wall = time.perf_counter() - t1

    n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n{len(PROMPTS)} prompts, {n_out} output tokens in {wall:.2f}s = {n_out / wall:.0f} tok/s")
    for i, out in enumerate(outputs):
        print(f"\n--- Prompt {i} ---")
        print(out.outputs[0].text[:200])


if __name__ == "__main__":
    main()
