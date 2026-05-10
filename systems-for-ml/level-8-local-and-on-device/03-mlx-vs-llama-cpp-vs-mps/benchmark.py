"""Three-way Apple Silicon LLM throughput bench: MLX vs llama.cpp Metal vs PyTorch MPS.

Produces G18 of Project 4. Greedy decoding, fixed prompt, fixed max_tokens.

The script tries all three backends. Missing ones print a notice and skip.

Setup before running:
  pip install mlx mlx-lm torch transformers llama-cpp-python
  # Pull MLX model
  huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit
  # Pull a GGUF for llama.cpp
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


PROMPT = (
    "Explain in one short paragraph what unified memory means on "
    "Apple Silicon and why it matters for large language model inference."
)
MAX_TOKENS = 256


@dataclass
class Result:
    backend: str
    ttft_s: float
    tokens: int
    decode_s: float

    @property
    def tok_per_s(self) -> float:
        return self.tokens / self.decode_s if self.decode_s else 0.0


def bench_mlx() -> Result | None:
    try:
        from mlx_lm import load, stream_generate
    except ImportError:
        print("[mlx] mlx_lm not installed; skip")
        return None

    print("[mlx] loading...")
    model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    msgs = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

    t0 = time.perf_counter()
    ttft = None
    n = 0
    for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS):
        if ttft is None:
            ttft = time.perf_counter() - t0
            t_decode_start = time.perf_counter()
        n += 1
    decode_s = time.perf_counter() - t_decode_start
    return Result("mlx-lm", ttft or 0.0, n, decode_s)


def bench_llama_cpp() -> Result | None:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[llama.cpp] llama-cpp-python not installed; skip")
        return None

    gguf = os.environ.get(
        "GGUF_PATH",
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-GGUF/"
            "snapshots/main/qwen2.5-7b-instruct-q4_k_m.gguf"
        ),
    )
    if not os.path.exists(gguf):
        print(f"[llama.cpp] no GGUF at {gguf}; set GGUF_PATH env var")
        return None

    print("[llama.cpp] loading (n_gpu_layers=-1 for full Metal offload)...")
    llm = Llama(model_path=gguf, n_gpu_layers=-1, n_ctx=4096, verbose=False)

    t0 = time.perf_counter()
    ttft = None
    n = 0
    t_decode_start = t0
    for chunk in llm(
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n",
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stream=True,
    ):
        if ttft is None:
            ttft = time.perf_counter() - t0
            t_decode_start = time.perf_counter()
        n += 1
    decode_s = time.perf_counter() - t_decode_start
    return Result("llama.cpp", ttft or 0.0, n, decode_s)


def bench_torch_mps() -> Result | None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from threading import Thread
    except ImportError:
        print("[mps] torch/transformers not installed; skip")
        return None

    if not torch.backends.mps.is_available():
        print("[mps] MPS not available; skip")
        return None

    name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[mps] loading {name} in fp16 (PyTorch MPS, no 4-bit equivalent)...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16
    ).to("mps")

    msgs = [{"role": "user", "content": PROMPT}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to("mps")

    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    gen_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
        streamer=streamer,
    )

    t0 = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    ttft = None
    n = 0
    t_decode_start = t0
    for _ in streamer:
        if ttft is None:
            ttft = time.perf_counter() - t0
            t_decode_start = time.perf_counter()
        n += 1
    thread.join()
    decode_s = time.perf_counter() - t_decode_start
    return Result("torch-mps", ttft or 0.0, n, decode_s)


def main():
    results: list[Result] = []
    for fn in (bench_mlx, bench_llama_cpp, bench_torch_mps):
        r = fn()
        if r is not None:
            results.append(r)
            print(
                f"  -> {r.backend:12s} TTFT={r.ttft_s*1000:6.0f} ms  "
                f"decode={r.tok_per_s:6.1f} tok/s  ({r.tokens} tokens)"
            )

    print("\n=== summary ===")
    print(f"{'backend':12s} {'TTFT (ms)':>10s} {'tok/s':>8s}")
    for r in results:
        print(f"{r.backend:12s} {r.ttft_s*1000:10.0f} {r.tok_per_s:8.1f}")


if __name__ == "__main__":
    main()
