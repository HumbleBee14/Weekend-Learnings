"""
MoE vs Dense — measure prefill tok/s, decode tok/s, and peak RAM on Apple
Silicon via mlx_lm.

The point: at matched 4-bit, an MoE often decodes faster than a smaller
dense model because per-token bandwidth scales with *active* params, not
total. Total RAM still scales with total params.
"""
from __future__ import annotations
import argparse
import resource
import time

from mlx_lm import load, generate
import mlx.core as mx


def _peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: Darwin reports bytes.
    return rss / (1024 ** 3) if rss > 1e8 else rss / (1024 ** 2)


def run_one(model_id: str, prompt: str, max_tokens: int) -> dict:
    print(f"\nLoading {model_id}...")
    model, tokenizer = load(model_id)

    # Warmup (also primes any lazy MLX kernels and unified-memory pages).
    _ = generate(model, tokenizer, prompt="hello", max_tokens=4, verbose=False)

    n_prompt = len(tokenizer.encode(prompt))

    # Prefill: time the very first token.
    t0 = time.perf_counter()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    mx.metal.synchronize() if hasattr(mx, "metal") else None
    t_prefill = time.perf_counter() - t0
    prefill_tps = n_prompt / t_prefill if t_prefill > 0 else float("inf")

    # Decode: full generation, then back out the per-decoded-token rate.
    t0 = time.perf_counter()
    out = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, verbose=False
    )
    t_total = time.perf_counter() - t0
    n_out = len(tokenizer.encode(out)) - n_prompt
    decode_tps = n_out / max(t_total - t_prefill, 1e-6)

    return {
        "model": model_id,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "peak_rss_gb": _peak_rss_gb(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dense", required=True)
    p.add_argument("--moe", required=True)
    p.add_argument("--prompt", default="Explain MoE active-parameter routing in 200 words.")
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    results = []
    for kind, mid in (("Dense", args.dense), ("MoE", args.moe)):
        r = run_one(mid, args.prompt, args.max_tokens)
        r["kind"] = kind
        results.append(r)

    for r in results:
        print(f"\n=== {r['kind']:5s} {r['model']} ===")
        print(f"prefill tok/s: ~{r['prefill_tps']:.0f}")
        print(f" decode tok/s: ~{r['decode_tps']:.0f}")
        print(f"   peak RAM:   {r['peak_rss_gb']:.1f} GB")

    if len(results) == 2:
        d = (results[1]["decode_tps"] / results[0]["decode_tps"] - 1) * 100
        print(f"\nDecode delta:  {d:+.0f}% (MoE vs Dense)")


if __name__ == "__main__":
    main()
