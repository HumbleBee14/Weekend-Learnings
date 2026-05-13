"""Demonstrate recompilation pain and the three fixes.

Run each mode separately:
    TORCH_LOGS=recompiles python recompile_demo.py --mode naive
    TORCH_LOGS=recompiles python recompile_demo.py --mode mark_dynamic
    TORCH_LOGS=recompiles python recompile_demo.py --mode dynamic
    TORCH_LOGS=recompiles python recompile_demo.py --mode bucket
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)))


def time_call(fn, x, n: int = 20) -> float:
    # warmup
    for _ in range(3):
        y = fn(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y = fn(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n  # ms per call


def bucket(seqlen: int, buckets: list[int]) -> int:
    for b in buckets:
        if seqlen <= b:
            return b
    return buckets[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["naive", "mark_dynamic", "dynamic", "bucket"], default="naive")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--seqlens", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128, 192, 256, 384])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(0)
    mlp = TinyMLP(dim=args.dim).to(device=device, dtype=dtype).eval()

    # In bucket mode we pad to a small set of fixed shapes.
    buckets = [16, 32, 64, 128, 256, 512]

    @torch.inference_mode()
    def run(seqlen: int) -> float:
        x = torch.randn(1, seqlen, args.dim, device=device, dtype=dtype)

        if args.mode == "naive":
            compiled = torch.compile(mlp)
        elif args.mode == "mark_dynamic":
            torch._dynamo.mark_dynamic(x, 1)
            compiled = torch.compile(mlp)
        elif args.mode == "dynamic":
            compiled = torch.compile(mlp, dynamic=True)
        elif args.mode == "bucket":
            b = bucket(seqlen, buckets)
            pad = b - seqlen
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            compiled = run.compiled_buckets.setdefault(b, torch.compile(mlp))

        return time_call(compiled, x, n=20)

    run.compiled_buckets = {}  # only used in bucket mode

    print(f"mode={args.mode}, device={device}, dim={args.dim}")
    print(f"{'seqlen':>8s}  {'ms/iter':>10s}")
    for s in args.seqlens:
        ms = run(s)
        print(f"{s:8d}  {ms:10.3f}")

    # Print cache info
    try:
        # cache size on the compiled module
        cache = torch._dynamo.utils.compile_times()
        print()
        print("compile times by frame (approx, may be empty if no recompiles):")
        print(cache)
    except Exception:
        pass

    print()
    print("Run with TORCH_LOGS=recompiles to see the recompile reasons inline.")


if __name__ == "__main__":
    main()
