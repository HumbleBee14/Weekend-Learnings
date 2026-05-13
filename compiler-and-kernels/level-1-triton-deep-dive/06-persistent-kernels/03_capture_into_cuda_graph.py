"""
03 — Capture the persistent matmul into a CUDA graph.

This is where the persistent pattern pays off. With a fixed grid of (num_SMs,),
the kernel launch parameters are independent of M/N/K. We can record one
torch.cuda.graph that handles any decode-shape matmul through the same
pre-allocated buffers; replay costs one driver round-trip instead of one
per kernel.

The exercise:
  1. Pre-allocate fixed input/output buffers.
  2. Warm up so autotune (if any) and the JIT compile happen outside capture.
  3. Capture the persistent matmul launch into a torch.cuda.CUDAGraph.
  4. Replay it and compare against eager launch of the same kernel at M=1.

Expected on T4 at M=1 N=4096 K=4096:
  eager persistent launch: ~40-80 us (kernel + ~7us launch overhead)
  graph replay           : ~10-30 us (~5us replay submit + kernel)
  => 2-5x speedup, almost all of it eliminating the launch.

Reference: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
          https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
"""

import torch
import triton
import triton.language as tl

# Reuse the persistent kernel from file 01. We re-paste it here so this file
# stands alone — Colab users running just this script don't need to import
# across files.

@triton.jit
def persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_K < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# The launch helper: takes *the same buffers every call*. This is what makes
# the graph capture work — the captured launch records these exact data_ptrs.
def launch_persistent_matmul(a, b, c, num_sms, BLOCK_M=16, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    _, N = b.shape
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, total_tiles),)
    persistent_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=grid[0],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_properties("cuda")
    num_sms = dev.multi_processor_count
    print(f"GPU: {dev.name}, {num_sms} SMs")

    torch.manual_seed(0)
    # Decode-ish shape: one query token, hidden dim 4096, K=4096. This is
    # roughly what every linear layer of a LLaMA-7B looks like at decode.
    M, N, K = 1, 4096, 4096
    dtype = torch.float16

    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    c = torch.empty((M, N), device="cuda", dtype=dtype)

    # ---- Warmup. JITs the kernel, picks any best config, populates caches. ----
    # The Triton tutorials emphasize: never capture without a warmup. The
    # first call compiles the kernel and may launch internal benchmark configs.
    for _ in range(5):
        launch_persistent_matmul(a, b, c, num_sms)
    torch.cuda.synchronize()

    # Correctness baseline.
    c_ref = (a.float() @ b.float()).to(dtype)
    err = (c.float() - c_ref.float()).abs().max().item()
    print(f"eager correctness: max diff = {err:.2e}")

    # ---- Eager launch timing ----
    ms_eager = triton.testing.do_bench(lambda: launch_persistent_matmul(a, b, c, num_sms),
                                       warmup=25, rep=200)
    print(f"\neager persistent launch: {ms_eager*1000:.2f} us")

    # ---- CUDA graph capture ----
    # Canonical idiom: capture happens on a side stream. We allocate the graph
    # object, enter the capture context, perform the launches we want recorded.
    # Inside the context, every CUDA op goes into the graph instead of executing.
    #
    # Pitfall: torch.cuda.graph internally uses a side stream. If you launch on
    # the default stream during capture, capture fails. The context manager
    # handles this for you — just don't manually .cuda() new tensors inside.

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        # One more warmup on the side stream — Triton's launch path may stash
        # some state per-stream the first time it sees one.
        for _ in range(3):
            launch_persistent_matmul(a, b, c, num_sms)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch_persistent_matmul(a, b, c, num_sms)

    # Replay returns immediately on host; the device runs the captured stream.
    g.replay()
    torch.cuda.synchronize()
    err_graph = (c.float() - c_ref.float()).abs().max().item()
    print(f"graph replay correctness: max diff = {err_graph:.2e}")

    # Time the replay. do_bench is happy to time arbitrary callables.
    ms_graph = triton.testing.do_bench(lambda: g.replay(), warmup=25, rep=500)
    print(f"graph replay           : {ms_graph*1000:.2f} us")
    print(f"speedup: {ms_eager/ms_graph:.2f}x")

    # ---- Shape-stability sanity check ----
    # The graph captured M=1. If we change M (allocate a new `a`), the captured
    # launch still uses the old data pointer — this would silently use stale
    # data. The vLLM solution: pre-allocate one giant buffer, copy real inputs
    # into it before replay. We demonstrate the *correct* pattern.
    print("\nCorrect-use pattern: mutate the captured buffers in place, then replay.")
    a.copy_(torch.randn_like(a))  # In-place write, same data_ptr.
    g.replay()
    torch.cuda.synchronize()
    c_ref2 = (a.float() @ b.float()).to(dtype)
    err2 = (c.float() - c_ref2.float()).abs().max().item()
    print(f"after in-place input mutation + replay: max diff = {err2:.2e}")

    # ---- The wrong pattern, for illustration ----
    # If you allocate a new tensor, the graph still references the old one.
    a_new = torch.randn_like(a)  # different data_ptr
    g.replay()
    torch.cuda.synchronize()
    c_ref_new = (a_new.float() @ b.float()).to(dtype)
    err_wrong = (c.float() - c_ref_new.float()).abs().max().item()
    print(f"after fresh allocation + replay (incorrect use): max diff = {err_wrong:.2e}")
    print("  ^ This should be LARGE — the graph used the old buffer. This is the")
    print("    most common CUDA-graph footgun. Pre-allocate and copy_ into the buffer.")

    print("\nLesson: the 2-5x decode speedup is real but only if you respect the")
    print("captured-pointer rule. vLLM's CUDAGraphDispatcher pre-allocates one")
    print("input buffer per shape bucket and copies real inputs in before replay.")


if __name__ == "__main__":
    main()
