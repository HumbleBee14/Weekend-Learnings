"""
03 — Tiled matmul with TMA + autotune + early_config_prune.

This is the keeper. Production-shaped: a real configuration space, a real
pruning function, and `tl.make_tensor_descriptor` for the load path. The
first call autotunes (takes 30-60 seconds typically); subsequent calls reuse
the winner per shape.

Expected on H100: 75-95% of torch.matmul on 4096^3 fp16.
Expected on T4:   60-85% of torch.matmul on 4096^3 fp16.

Read the winning config printed at the bottom. Write three sentences in
notes.md explaining why that config won — not which it was, but why.
"""

import torch
import triton
import triton.language as tl


def _gen_configs():
    cfgs = []
    for bm in [64, 128, 256]:
        for bn in [64, 128, 256]:
            for bk in [32, 64, 128]:
                for nw in [4, 8]:
                    for ns in [2, 3, 4, 5]:
                        cfgs.append(triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=nw, num_stages=ns,
                        ))
    return cfgs


def _prune_configs(configs, named_args, **kwargs):
    """Filter configs that won't compile or are obvious losers."""
    M, N, K = named_args["M"], named_args["N"], named_args["K"]
    out = []
    for cfg in configs:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        nw = cfg.num_warps
        ns = cfg.num_stages

        # Need at least some output tiles
        if bm > M or bn > N:
            continue

        # SRAM budget for the pipelined A+B tiles. fp16 = 2 bytes.
        smem_bytes = ns * (bm * bk + bk * bn) * 2
        if smem_bytes > 200 * 1024:  # leave some room under H100's 228KB; T4 has ~96KB
            continue

        # Register pressure for the accumulator. fp32 = 4 bytes.
        # `acc` is BLOCK_M * BLOCK_N fp32 spread across (nw * 32) lanes.
        acc_bytes_per_lane = (bm * bn * 4) // (nw * 32)
        if acc_bytes_per_lane > 256:  # 256 bytes per lane is the practical register ceiling
            continue

        # Tensor-core fragment alignment. Fragments are 16x8x16 minimum on most archs.
        if bm < 16 or bn < 16 or bk < 16:
            continue

        out.append(cfg)
    return out


@triton.autotune(
    configs=_gen_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def matmul_auto_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    for k in range(0, K, BLOCK_K):
        a = a_desc.load([offs_m, k])
        b = b_desc.load([k, offs_n])
        acc += tl.dot(a, b)

    c_desc.store([offs_m, offs_n], acc.to(c_ptr.dtype.element_ty))


def matmul_auto(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_auto_kernel[grid](a, b, c, M, N, K)
    return c


def main():
    torch.manual_seed(0)
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    print("Autotuning (first call) — this takes 30-60s...")
    c = matmul_auto(a, b)
    diff = (c - a @ b).abs().max().item()
    print(f"correctness: max diff = {diff:.2e}")

    print(f"\nBest config: {matmul_auto_kernel.best_config}")

    ms_triton = triton.testing.do_bench(lambda: matmul_auto(a, b))
    ms_torch = triton.testing.do_bench(lambda: a @ b)
    flops = 2 * M * N * K
    tflops_triton = flops / (ms_triton * 1e-3) / 1e12
    tflops_torch = flops / (ms_torch * 1e-3) / 1e12

    print(f"\n4096^3 fp16 matmul (autotuned):")
    print(f"  triton: {ms_triton:.2f} ms   {tflops_triton:.1f} TFLOPS   "
          f"{tflops_triton/tflops_torch*100:.0f}% of torch")
    print(f"  torch : {ms_torch:.2f} ms   {tflops_torch:.1f} TFLOPS")
    print()
    print("Look at the best config and explain (in notes.md) why this specific")
    print("(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) won on your hardware.")


if __name__ == "__main__":
    main()
