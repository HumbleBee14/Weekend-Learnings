"""
Compare your hand-written autotuned matmul against torch.compile-emitted Triton.

What you should see:
  - On a "common" shape like 4096^3, Inductor and your kernel will be within 5-10%
    of each other, both within 10% of cuBLAS.
  - On an "unusual" shape (small M, ragged dims), one will sometimes win significantly.
    Read why.

Save the table and your interpretation to notes.md.
"""

import importlib.util, os, torch, triton

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("autotuned", os.path.join(HERE, "03_autotuned.py"))
autotuned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autotuned)


@torch.compile(mode="max-autotune", fullgraph=True)
def compiled_matmul(a, b):
    return a @ b


def main():
    torch.manual_seed(0)

    # Warm up torch.compile outside timing
    _ = compiled_matmul(torch.randn(64, 64, device="cuda", dtype=torch.float16),
                       torch.randn(64, 64, device="cuda", dtype=torch.float16))

    shapes = [
        (4096, 4096, 4096),    # square, common
        (8192, 8192, 8192),    # square, larger
        (1, 4096, 4096),       # decode (batch 1, single token)
        (32, 4096, 11008),     # MLP up-projection at hidden=4096, intermediate=11008
        (32, 11008, 4096),     # MLP down-projection
    ]

    print(f"{'shape':<25}{'cuBLAS':>10}{'auto':>10}{'compile':>10}{'auto/cuBLAS':>14}{'compile/cuBLAS':>16}")
    print("-" * 85)
    for M, K, N in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        ms_torch = triton.testing.do_bench(lambda: a @ b)
        ms_auto = triton.testing.do_bench(lambda: autotuned.matmul_auto(a, b))
        ms_compile = triton.testing.do_bench(lambda: compiled_matmul(a, b))

        flops = 2 * M * N * K
        t_torch = flops / (ms_torch * 1e-3) / 1e12
        t_auto = flops / (ms_auto * 1e-3) / 1e12
        t_compile = flops / (ms_compile * 1e-3) / 1e12

        print(f"M={M} K={K} N={N:<8}{t_torch:>9.1f}T{t_auto:>9.1f}T{t_compile:>9.1f}T"
              f"{t_auto/t_torch*100:>13.0f}%{t_compile/t_torch*100:>15.0f}%")

    print()
    print("Note: the 'decode' shape (M=1) is memory-bound, not compute-bound — the metric")
    print("'% of peak TFLOPS' is misleading there. The right metric for M=1 is bandwidth.")
    print("That's why production engines use a different kernel for decode (sub-module 06).")


if __name__ == "__main__":
    main()
