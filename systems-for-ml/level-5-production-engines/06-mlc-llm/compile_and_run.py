"""
06 - MLC-LLM minimal compile + chat loop.

This shows the compile-once-run-anywhere shape. On first run you compile
the model for your local target (CUDA / Metal / Vulkan); subsequent runs
load the cached artifact.

For most learners, the bigger value is opening https://chat.webllm.ai/ in
Chrome and watching a 7B run in the browser via WebGPU. That demo is the
clearest illustration of why MLC exists.

Prereqs:
    pip install --pre mlc-llm-nightly mlc-ai-nightly
        # nightly because MLC moves fast; pin in production

Run:
    python compile_and_run.py
"""

from __future__ import annotations

import time

try:
    from mlc_llm import MLCEngine  # type: ignore
except ImportError:
    print("mlc_llm not installed — `pip install --pre mlc-llm-nightly mlc-ai-nightly`")
    raise SystemExit(0)


# Pre-compiled artifacts MLC publishes — saves you the multi-minute compile.
# See https://huggingface.co/mlc-ai for the catalog.
MODEL = "HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC"


def main() -> None:
    print(f"Loading {MODEL} (first run downloads + JITs the kernels)...")
    t0 = time.perf_counter()
    engine = MLCEngine(MODEL)
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    prompts = [
        "Why does MLC-LLM exist when vLLM and llama.cpp already do?",
        "Explain WebGPU in one sentence.",
        "When would I pick TVM over a hand-tuned kernel library?",
    ]

    t1 = time.perf_counter()
    n_tokens = 0
    for p in prompts:
        print(f"\n>>> {p}")
        for chunk in engine.chat.completions.create(
            messages=[{"role": "user", "content": p}],
            model=MODEL,
            stream=True,
            max_tokens=128,
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                n_tokens += 1
        print()
    wall = time.perf_counter() - t1
    print(f"\nTotal: ~{n_tokens} tokens in {wall:.1f}s = {n_tokens / wall:.0f} tok/s")
    engine.terminate()


if __name__ == "__main__":
    main()
