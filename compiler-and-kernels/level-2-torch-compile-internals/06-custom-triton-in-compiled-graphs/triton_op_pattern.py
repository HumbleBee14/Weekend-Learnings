"""Pattern A — torch.library.triton_op.

Inductor can trace through this wrapper. Epilogue fusion works.
"""

from __future__ import annotations

import torch
from torch.library import triton_op, wrap_triton

from rmsnorm_kernel import rmsnorm_kernel, next_pow2


@triton_op("level2::rmsnorm_triton_op", mutates_args={})
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = next_pow2(N)
    wrap_triton(rmsnorm_kernel)[(M,)](
        x, weight, out,
        x.stride(0), out.stride(0),
        N=N, eps=eps, BLOCK_SIZE=BLOCK,
    )
    return out


def main() -> None:
    assert torch.cuda.is_available()
    dtype = torch.bfloat16
    M, N = 2048, 4096
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.ones(N, device="cuda", dtype=dtype)

    # Verify against eager
    y = rmsnorm(x, w)
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    y_ref = (x.float() * rms * w.float()).to(dtype)
    print(f"max abs err: {(y - y_ref).abs().max().item():.4e}")

    # Compose with torch.compile + residual add (the epilogue to fuse)
    def block(x, w, residual):
        return rmsnorm(x, w, 1e-6) + residual

    compiled = torch.compile(block, fullgraph=True)
    residual = torch.randn_like(x)
    for _ in range(3):
        out = compiled(x, w, residual)
    print(f"compiled mean: {out.float().mean().item():.4f}")
    print()
    print("Now run with TORCH_COMPILE_DEBUG=1 and inspect output_code.py.")
    print("You should see the residual add inside the same kernel as rmsnorm.")


if __name__ == "__main__":
    main()
