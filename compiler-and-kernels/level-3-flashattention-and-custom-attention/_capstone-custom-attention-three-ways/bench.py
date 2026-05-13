"""Head-to-head benchmark of the three capstone implementations.

Run on A100:
    python bench.py
"""
from __future__ import annotations

import torch
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel


def bench_one(name: str, fn, warmup=25, rep=100) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    from impl_triton import capstone_attention_triton
    from impl_flex import capstone_attention_flex, _alibi_slopes
    try:
        from impl_flashinfer import capstone_attention_flashinfer
        have_fi = True
    except Exception:
        have_fi = False

    torch.manual_seed(0)
    B, H, D = 2, 8, 64
    W, S = 512, 4

    print(f"{'N':>6} {'sdpa(full)':>14} {'triton':>10} {'flex':>10} {'flashinfer':>12}")
    for N in [2048, 4096, 8192, 16384]:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
        slopes = torch.tensor(_alibi_slopes(H), device="cuda", dtype=torch.float32)

        # SDPA causal (no ALiBi, no window — just a baseline).
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                ms_sdpa = bench_one("sdpa",
                    lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True))
        except Exception:
            ms_sdpa = float("nan")

        ms_tri = bench_one("triton",
            lambda: capstone_attention_triton(q, k, v, window=W, sinks=S, slopes=slopes))
        ms_flex = bench_one("flex",
            lambda: capstone_attention_flex(q, k, v, window=W, sinks=S))

        if have_fi and B == 1:
            ms_fi = bench_one("fi",
                lambda: capstone_attention_flashinfer(q[:1], k[:1], v[:1], window=W, sinks=S))
        else:
            ms_fi = float("nan")

        print(f"{N:>6} {ms_sdpa:>14.3f} {ms_tri:>10.3f} {ms_flex:>10.3f} {ms_fi:>12.3f}")

    print("\nDrop these into report.md alongside your writeup.")


if __name__ == "__main__":
    main()
