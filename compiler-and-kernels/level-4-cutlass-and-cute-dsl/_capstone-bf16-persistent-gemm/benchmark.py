"""
benchmark.py — run the capstone benchmark grid and emit the report table.

Compares:
  - cuBLAS (via torch.matmul)
  - Your Triton matmul from compiler-and-kernels/level-1 (if importable)
  - The five stages from compiler-and-kernels/level-4/04-* (if importable)
  - This capstone's `gemm.matmul` (the tuned persistent kernel)

Shapes:
  - Square: 512, 1024, 2048, 4096, 8192
  - LLaMA-7B FFN-1: M=8192, K=4096, N=11008
  - LLaMA-7B QKV combined: M=8192, K=4096, N=12288
  - Decode: M=8, K=4096, N=4096

Output: reports/benchmark_<hostname>_<date>.md
"""

import datetime
import os
import socket
import torch

import gemm


SHAPES = [
    ("square_512",       512, 512, 512),
    ("square_1024",      1024, 1024, 1024),
    ("square_2048",      2048, 2048, 2048),
    ("square_4096",      4096, 4096, 4096),
    ("square_8192",      8192, 8192, 8192),
    ("llama_ffn1",       8192, 11008, 4096),
    ("llama_qkv",        8192, 12288, 4096),
    ("decode_M8",        8, 4096, 4096),
]


def bench(fn, n_iter: int = 100, n_warmup: int = 25) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter


def tflops(M: int, N: int, K: int, ms: float) -> float:
    return (2.0 * M * N * K) / (ms * 1e9)


def run_shape(name: str, M: int, N: int, K: int) -> dict:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    row = {"name": name, "M": M, "N": N, "K": K}

    # cuBLAS via torch.matmul.
    ms_cublas = bench(lambda: torch.matmul(a, b))
    row["cublas_tflops"] = tflops(M, N, K, ms_cublas)
    row["cublas_ms"] = ms_cublas

    # Try to import the level-1 Triton matmul.
    try:
        import sys
        sys.path.append(os.path.join(
            os.path.dirname(__file__), "..", "..",
            "level-1-triton-deep-dive", "04-tiled-matmul-and-autotune",
        ))
        from triton_matmul import matmul as triton_matmul  # type: ignore
        ms_triton = bench(lambda: triton_matmul(a, b))
        row["triton_tflops"] = tflops(M, N, K, ms_triton)
    except ImportError:
        row["triton_tflops"] = None

    # Try the capstone GEMM (may fail on non-tile-aligned shapes).
    try:
        ms_capstone = bench(lambda: gemm.matmul(a, b))
        row["capstone_tflops"] = tflops(M, N, K, ms_capstone)
    except Exception as exc:
        print(f"  capstone failed for {name}: {exc}")
        row["capstone_tflops"] = None

    return row


def format_table(rows: list[dict]) -> str:
    out = [
        "| Shape | (M, N, K) | cuBLAS TFLOPS | Triton (L1) TFLOPS | Capstone TFLOPS | % cuBLAS |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        cu = r["cublas_tflops"]
        tr = r.get("triton_tflops")
        ca = r.get("capstone_tflops")
        pct = f"{100.0 * ca / cu:.1f}%" if ca else "—"
        out.append(
            f"| {r['name']} | ({r['M']},{r['N']},{r['K']}) | "
            f"{cu:.1f} | "
            f"{tr:.1f}" if tr else "—" + " | "
            f"{ca:.1f}" if ca else "—" + f" | {pct} |"
        )
    return "\n".join(out)


def main():
    rows = []
    for name, M, N, K in SHAPES:
        print(f"running {name} ({M},{N},{K}) ...")
        rows.append(run_shape(name, M, N, K))

    table = format_table(rows)
    print()
    print(table)

    os.makedirs("reports", exist_ok=True)
    host = socket.gethostname().replace(" ", "_")
    date = datetime.date.today().isoformat()
    path = f"reports/benchmark_{host}_{date}.md"
    with open(path, "w") as f:
        f.write(f"# Benchmark — {host} {date}\n\n")
        gpu = torch.cuda.get_device_name(0)
        f.write(f"GPU: {gpu}\n")
        f.write(f"CUDA: {torch.version.cuda}\n")
        f.write(f"Torch: {torch.__version__}\n\n")
        f.write(table)
        f.write("\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
