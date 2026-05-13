# 06 — Custom Triton kernels inside a compiled graph

You take a Triton RMSNorm kernel (your Level 1 sub-module 03 kernel, or the bundled one in [`rmsnorm_kernel.py`](rmsnorm_kernel.py)) and make it compose with `torch.compile`. Two patterns, both implemented:

- **Pattern A: `triton_op`** — Inductor can trace into your op, fuse epilogues onto it. Recommended.
- **Pattern B: `custom_op` + `register_fake`** — Inductor treats your op as a black box. Older pattern; included so you understand the cost of opacity.

You measure the difference. On bandwidth-bound shapes, Pattern A wins by 5–15% because Inductor can fold a downstream residual add or activation into your kernel's epilogue.

## Hardware

T4 (or any CUDA GPU). Custom Triton kernels need CUDA.

## What to run

```bash
python triton_op_pattern.py     # Pattern A: triton_op (recommended)
python custom_op_pattern.py     # Pattern B: custom_op + register_fake
python compare.py               # benchmark both inside a compiled block
```

[`compare.py`](compare.py) wraps each pattern in a `RMSNorm + (residual add)` block and times both inside `torch.compile`. The residual add is the epilogue that Pattern A can fuse and Pattern B cannot.

## What you should observe

Steady-state per-iter, bf16, hidden=4096, seqlen=2048 on T4 (your numbers will vary):

| Variant | ms/iter | Notes |
|---|---|---|
| Eager (torch RMSNorm + add) | ~0.55 | reference |
| Pattern A (`triton_op`) inside `torch.compile` | ~0.36 | residual add fused into kernel |
| Pattern B (`custom_op`) inside `torch.compile` | ~0.42 | residual add is a separate kernel |

Verify with depyf (output dir set in each script) that Pattern A produces one FX node Inductor can see into, and Pattern B produces one opaque node with a separate add following it.

## What's in each pattern

**`triton_op`** wraps a Python function that calls your Triton kernel. Inductor knows about the wrapper: it can autotune, it can fuse, and it composes with `torch.export`. The wrapper still uses `@triton.jit` under the hood — nothing about your kernel changes, only the way it's exposed.

```python
from torch.library import triton_op, wrap_triton

@triton_op("mylib::rmsnorm", mutates_args={})
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = torch.empty_like(x)
    n_rows = x.shape[0]
    wrap_triton(rmsnorm_kernel)[(n_rows,)](x, weight, out, ...)
    return out
```

**`custom_op` + `register_fake`** wraps a function that calls your kernel but is opaque to Inductor:

```python
@torch.library.custom_op("mylib::rmsnorm_opaque", mutates_args=())
def rmsnorm_opaque(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # same implementation
    ...

@torch.library.register_fake("mylib::rmsnorm_opaque")
def _(x, weight, eps=1e-6):
    return torch.empty_like(x)
```

The `register_fake` ("meta kernel") tells Dynamo what shape/dtype to expect without running the op. Required because Dynamo traces with `FakeTensor`s that have no data.

## Reference

- [PyTorch tutorial — user-defined Triton kernels with torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html).
- [`torch.library` docs](https://docs.pytorch.org/docs/stable/library.html).
- [Inductor Triton Custom Op — dev discussion](https://dev-discuss.pytorch.org/t/inductor-triton-custom-op/1704).

## Common pitfalls

- **You wrote `@torch.library.custom_op` and got "no fake impl found".** Register one with `@torch.library.register_fake`. The fake impl returns empty tensors with the right shapes; no actual computation.
- **Your `triton_op`-wrapped kernel still doesn't fuse.** Check the FX graph — Inductor only fuses when the iteration spaces match. A row-reduction followed by a row-broadcast pointwise add should fuse; a row-reduction followed by a column op will not.
- **You mutate an input tensor and `mutates_args={}` lied.** Declare the mutation: `mutates_args=("out",)`. Otherwise Dynamo's functionalization will assume your op is pure and downstream reads will see stale data.
- **You wrote a backward pass and Inductor refuses to compile.** Custom ops need `register_autograd` for training. Inference-only is much simpler — we stay inference-only here.
