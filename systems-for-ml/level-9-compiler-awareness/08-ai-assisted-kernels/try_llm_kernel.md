# Exercise — prompt an LLM to write a Triton softmax

The point isn't to ship a kernel. It's to feel the loop: prompt → output → compile → numerics check → benchmark → re-prompt. That's the harness AI-assisted kernel work lives inside.

## Setup

```bash
pip install torch triton openai anthropic  # pick your provider
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
```

A GPU helps for the benchmark step. CPU users: skip the bench, do the correctness check via `torch.testing.assert_close` and read the recorded transcript at the bottom.

## The prompt

```
Write a Triton kernel that computes row-wise softmax for a 2D tensor of shape (M, N).
Requirements:
- Input dtype fp16 or fp32, output same dtype.
- Numerically stable (subtract max before exp).
- Use BLOCK_SIZE: tl.constexpr for N, assume N <= 4096.
- Provide a Python wrapper softmax(x: torch.Tensor) -> torch.Tensor that
  launches the kernel with a 1D grid of M.

Return only the code, no commentary.
```

## Harness

```python
import torch
import torch.nn.functional as F

def check(softmax_fn, M=256, N=1024, dtype=torch.float32, device='cuda'):
    x = torch.randn(M, N, dtype=dtype, device=device)
    got = softmax_fn(x)
    want = F.softmax(x, dim=-1)
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)
    print("numerics ok")

def bench(softmax_fn, M=4096, N=4096, dtype=torch.float16, device='cuda', iters=200):
    import time
    x = torch.randn(M, N, dtype=dtype, device=device)
    # warmup
    for _ in range(20):
        softmax_fn(x); F.softmax(x, dim=-1)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        softmax_fn(x)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - t0) * 1000 / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        F.softmax(x, dim=-1)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000 / iters

    print(f"triton {triton_ms:.3f} ms  torch {torch_ms:.3f} ms  ratio {torch_ms/triton_ms:.2f}x")
```

## What to expect

First-shot output, frontier model, 2026:

- Correct numerics: usually yes for fp32. fp16 sometimes fails the `atol=1e-3` because the model forgot to do the max-subtract in fp32.
- Compiles: usually yes; occasional errors around `tl.constexpr` or grid-launch shape.
- Faster than `F.softmax`: rarely on first try, usually within 1.5x. With one round of "your kernel is 1.4x slower than torch on (4096, 4096) fp16, what could you change?" it often closes the gap.

The interesting failure modes:

- **Forgets the boundary mask** when N is not a power of two. Wraps reads off the end of the row, producing garbage in the last block's output.
- **Stores in fp32 then casts to fp16** correctly, but allocates a fp32 output buffer and forgets to cast it back. Numerics pass, dtype check fails.
- **Picks BLOCK_SIZE = N** with no autotune, so for N=4096 you get one block per row; fine on H100, slow on smaller GPUs.

## Recorded transcript (no-GPU readers)

If you don't have a GPU, the qualitative result of the above on H100 with a frontier model in early 2026:

```
Round 1 (zero-shot):
  numerics ok
  triton 0.142 ms  torch 0.098 ms  ratio 0.69x

Round 2 (after "you're slower than torch, what would you change?"):
  numerics ok
  triton 0.091 ms  torch 0.098 ms  ratio 1.08x

Round 3 (asked to tune BLOCK_SIZE per shape):
  numerics ok
  triton 0.082 ms  torch 0.098 ms  ratio 1.20x
```

That's three rounds. Each round is a few seconds of model time plus tens of seconds of compile-and-bench. On a tight harness, an hour of human time gets to "matches torch within 20%." Beating cuBLAS-class baselines (the matmul case) on first-shot prompting essentially never happens; with a multi-round agentic harness it does happen for some shapes, with the caveats covered in CONCEPTS.md.

## What to take away

- The LLM is one line of the harness. The interesting engineering is the harness.
- Frontier-model output is good enough to be a starting draft. It's not good enough to ship without review.
- Where LLMs win cleanly today: autotune-config proposals on existing kernels, and first-draft kernels for vanilla ops on well-known shapes.
- Where they don't: novel hardware targets, speed-of-light hand-tuned kernels, numerically subtle low-precision work.

That's the realistic shape of "AI-assisted kernels" in 2026.
