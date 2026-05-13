"""Pattern B — torch.library.custom_op + register_fake.

Inductor treats the op as a black box. Epilogue fusion does NOT happen.
"""

from __future__ import annotations

import torch

from rmsnorm_kernel import rmsnorm_kernel, next_pow2


@torch.library.custom_op("level2::rmsnorm_opaque", mutates_args=())
def rmsnorm_opaque(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = next_pow2(N)
    rmsnorm_kernel[(M,)](
        x, weight, out,
        x.stride(0), out.stride(0),
        N=N, eps=eps, BLOCK_SIZE=BLOCK,
    )
    return out


@torch.library.register_fake("level2::rmsnorm_opaque")
def _rmsnorm_opaque_fake(x, weight, eps=1e-6):
    # FakeTensor / meta impl: same shape and dtype, no data.
    return torch.empty_like(x)


def main() -> None:
    assert torch.cuda.is_available()
    dtype = torch.bfloat16
    M, N = 2048, 4096
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.ones(N, device="cuda", dtype=dtype)

    y = torch.ops.level2.rmsnorm_opaque(x, w)
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    y_ref = (x.float() * rms * w.float()).to(dtype)
    print(f"max abs err: {(y - y_ref).abs().max().item():.4e}")

    def block(x, w, residual):
        return torch.ops.level2.rmsnorm_opaque(x, w, 1e-6) + residual

    compiled = torch.compile(block, fullgraph=True)
    residual = torch.randn_like(x)
    for _ in range(3):
        out = compiled(x, w, residual)
    print(f"compiled mean: {out.float().mean().item():.4f}")
    print()
    print("Run with TORCH_COMPILE_DEBUG=1 — the residual add will be a SEPARATE kernel.")


if __name__ == "__main__":
    main()
