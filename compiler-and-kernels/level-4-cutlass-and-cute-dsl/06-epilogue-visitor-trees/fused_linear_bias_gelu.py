"""
fused_linear_bias_gelu.py — fuse bias add and GELU activation into the GEMM
epilogue using an Epilogue Visitor Tree.

  out = GELU(x @ W + b)

Pre-LLaMA-FFN-1 shape (M=batch*seq, K=hidden, N=intermediate). LLaMA-7B:
  M = 2048 * 32, K = 4096, N = 11008.

The fused version saves two full HBM round-trips of the (M, N) tile vs
unfused (matmul, then +bias, then gelu, three kernels).

The EVT in CuTe-DSL (current beta API — names may shift; the topology is
stable). The kernel-side mainloop is exactly stage5_persistent.py from
submodule 04; only the epilogue changes.

Run:
    python fused_linear_bias_gelu.py

Hardware: H100 or B200 (the current CuTe-DSL EVT path lowers cleanly on
SM90+).
"""

import math
import torch
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass.cute.epilogue import (
    AccumulatorSource, RowBroadcast, Compute, Store,
    EpilogueVisitorTree,
)
# Mainloop from submodule 04 stage 5
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04-tma-wgmma-and-persistent-gemm"))
from stage5_persistent import (   # type: ignore
    gemm_persistent_kernel,
    BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, TOTAL_WARPS,
)


def build_evt(bias_tma_desc, out_tma_desc):
    """Build: store(gelu(acc + bias))."""
    bias_node = RowBroadcast(bias_tma_desc, dtype=cutlass.Float32)
    add_node = Compute(op="plus", lhs=AccumulatorSource(), rhs=bias_node)
    gelu_node = Compute(op="gelu_tanh_approx", input=add_node)
    cast_node = Compute(op="cast", input=gelu_node, target=cutlass.BFloat16)
    store_node = Store(cast_node, out_tma_desc)
    return EpilogueVisitorTree(store_node)


# In practice the EVT plugs into the kernel via a `epilogue=` argument or
# a code-generation step. The exact wiring is API-specific; the upstream
# example to mimic is examples/python/CuTeDSL/hopper/dense_gemm_persistent.py
# combined with the EVT examples in examples/python/CuTeDSL/epilogue/.


def benchmark():
    M, K, N = 32 * 2048, 4096, 11008      # LLaMA-7B FFN-1
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    # Reference unfused.
    def unfused():
        return F.gelu(torch.matmul(x, W) + b, approximate="tanh")

    out_ref = unfused()

    # torch.compile reference.
    compiled = torch.compile(lambda x_, W_, b_: F.gelu(torch.matmul(x_, W_) + b_,
                                                       approximate="tanh"))
    out_compiled = compiled(x, W, b)
    assert torch.allclose(out_compiled, out_ref, atol=1e-2), "compile mismatch"

    # Your fused EVT kernel goes here. Wiring is left as an exercise:
    # take the stage5_persistent kernel and pass `epilogue=build_evt(...)`
    # in the @jit launcher.

    n_iter = 100
    def bench(fn):
        for _ in range(25):
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

    ms_unfused = bench(unfused)
    ms_compiled = bench(lambda: compiled(x, W, b))

    flops = 2.0 * M * N * K
    print(f"unfused        : {ms_unfused:7.3f} ms  {flops/(ms_unfused*1e9):6.1f} TFLOPS")
    print(f"torch.compile  : {ms_compiled:7.3f} ms  {flops/(ms_compiled*1e9):6.1f} TFLOPS")
    print("EVT fused      : (build the kernel and add a row)")


if __name__ == "__main__":
    benchmark()
