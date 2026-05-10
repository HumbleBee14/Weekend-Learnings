"""
KV cache quantization bench: same long prompt, fp16 vs 4-bit KV cache.
Reports prefill tok/s, decode tok/s, and approximate KV memory.

mlx_lm exposes kv_bits / kv_group_size in its sample/generate paths.
We use the load + generate API and pass the kv-cache flags.
"""
from __future__ import annotations
import argparse
import time

import mlx.core as mx
from mlx_lm import load, generate


FILLER = "Apple Silicon unified memory means CPU and GPU share DRAM. "


def synthesize_prompt(tokenizer, target_tokens: int) -> str:
    text = ""
    while len(tokenizer.encode(text)) < target_tokens:
        text += FILLER * 50
    enc = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(enc)


def kv_bytes_estimate(num_layers: int, num_kv_heads: int, head_dim: int,
                      seq_len: int, bits: int) -> float:
    # 2 (K and V) * layers * heads * head_dim * seq_len * (bits/8)
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * (bits / 8)


def run_one(model_id: str, prompt: str, max_tokens: int,
            kv_bits: int | None) -> dict:
    print(f"\nLoading {model_id}  kv_bits={kv_bits}...")
    model, tokenizer = load(model_id)
    n_prompt = len(tokenizer.encode(prompt))

    gen_kwargs = {"max_tokens": max_tokens, "verbose": False}
    if kv_bits is not None:
        gen_kwargs["kv_bits"] = kv_bits
        gen_kwargs["kv_group_size"] = 64
        gen_kwargs["quantized_kv_start"] = 0

    # warmup
    _ = generate(model, tokenizer, prompt="hi", max_tokens=2, verbose=False)

    t0 = time.perf_counter()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    if hasattr(mx, "metal"):
        mx.metal.synchronize()
    prefill_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
    total_t = time.perf_counter() - t0
    n_out = max(len(tokenizer.encode(out)) - n_prompt, 1)

    # Best-effort KV byte estimate from model config.
    cfg = getattr(model, "args", None) or getattr(model, "config", None)
    layers = getattr(cfg, "num_hidden_layers", 32)
    n_kv = getattr(cfg, "num_key_value_heads",
                   getattr(cfg, "num_attention_heads", 32))
    head_dim = getattr(cfg, "head_dim",
                       getattr(cfg, "hidden_size", 4096) // 32)
    bits = kv_bits if kv_bits else 16
    kv_b = kv_bytes_estimate(layers, n_kv, head_dim, n_prompt + n_out, bits)

    return {
        "kv_bits": bits,
        "prefill_tps": n_prompt / prefill_t,
        "decode_tps": n_out / max(total_t - prefill_t, 1e-6),
        "kv_gb": kv_b / (1024 ** 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--context-tokens", type=int, default=32000)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    _, tok = load(args.model)
    prompt = synthesize_prompt(tok, args.context_tokens)
    print(f"Prompt length: {len(tok.encode(prompt))} tokens")

    runs = []
    for kv in (None, 4):
        runs.append(run_one(args.model, prompt, args.max_tokens, kv))

    for r in runs:
        print(f"\n=== {r['kv_bits']}-bit KV ===")
        print(f"  prefill tok/s: ~{r['prefill_tps']:.0f}")
        print(f"  decode tok/s:  ~{r['decode_tps']:.0f}")
        print(f"  KV bytes:      ~{r['kv_gb']:.2f} GB")

    if len(runs) == 2 and runs[0]["kv_gb"] > 0:
        delta = (1 - runs[1]["kv_gb"] / runs[0]["kv_gb"]) * 100
        print(f"\nKV memory delta: {delta:.0f}% reduction.")


if __name__ == "__main__":
    main()
