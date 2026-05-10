# 02 — MLX Basics

## What MLX is

A NumPy-like + JAX-like array framework, native to Apple Silicon. Built by Apple's ML research org, open source under MIT. Three properties matter:

1. **Lazy graph.** Operations on `mx.array` build a computation graph; `mx.eval(...)` runs it. The compiler fuses what it can.
2. **Unified memory native.** No `to(device)`. Arrays live where every Apple Silicon array lives — in shared DRAM. The Metal backend and the CPU see the same buffers.
3. **NumPy-compatible-ish surface.** `mx.zeros`, `mx.matmul`, broadcasting, fancy indexing. Translating PyTorch or NumPy code is mostly a find-replace.

```
PyTorch:           x = torch.zeros(8, device='cuda'); y = (x + 1).sum()
                                    explicit copy, eager exec each line

MLX:               x = mx.zeros(8); y = (x + 1).sum()
                                    no device, graph builds, eval on demand
```

## Why lazy matters

Eager execution dispatches one Metal kernel per op. For a transformer layer that is dozens of tiny launches, each with a fixed overhead. Lazy execution lets MLX see the whole expression and:

- fuse elementwise ops into one kernel,
- keep intermediate tensors in registers/SRAM rather than writing through HBM,
- skip allocations for tensors that are never read.

The PyTorch path to the same property is `torch.compile`. `torch.compile` on MPS in 2026 is still partial — many ops fall back to eager. MLX is lazy by construction.

## The 2026 ecosystem

Top-level package is `mlx`. Application packages built on it:

- `mlx-lm` — text generation, fine-tuning, GGUF and safetensors loaders, KV cache quant, EAGLE-3 spec decode.
- `mlx-vlm` — vision-language: Qwen-VL, LLaVA, InternVL, Llama 4 vision.
- `mlx-whisper` — speech-to-text, faster than whisper.cpp on Apple Silicon.
- `mlx-embeddings` — local embedding inference; the missing piece for local RAG.
- `mlx-community` HuggingFace org — thousands of pre-quantized checkpoints in the MLX-native format.

Distributed MLX (April 2026) ships `mx.distributed`: ring all-reduce over Thunderbolt 5 between Macs. The basis for `exo`-style multi-Mac inference.

## Lazy graph in practice

```python
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0])
b = mx.array([10.0, 20.0, 30.0])
c = a * b + 1
# nothing has run yet. c is a node in the graph.
mx.eval(c)
# now the kernel runs and c holds [11, 41, 91].
```

Two consequences:

- **Print forces eval.** `print(c)` materializes. In hot loops, never `print` intermediate arrays.
- **Bench correctly.** Time `mx.eval(...)`, not the line that built the graph. Without `eval`, you are timing graph construction, which is microseconds and meaningless.

## Memory and dtypes

- `mx.float32`, `mx.float16`, `mx.bfloat16`, `mx.int32`, `mx.int8`, `mx.uint8`. No `int4`/`fp4` user-visible type — quantized models store packed bytes plus scales, decoded by op kernels.
- `mlx-lm` supports 2/3/4/6/8-bit weight-only quant. Quantize with `mlx_lm.convert`:

```bash
python -m mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ./qwen-7b-mlx-q4 \
  -q --q-bits 4 --q-group-size 64
```

Group size 64 is the 2026 default; 32 is sharper at slightly lower throughput.

## Autograd

`mx.grad(f)` returns the gradient function. Same shape as JAX. Used by `mlx-lm` for LoRA, full SFT, DPO. There is no separate autograd engine — gradients are graph transformations of the forward graph. This is why `mlx-lm` can fine-tune in surprisingly little code.

## Streams

`mx.default_stream(mx.gpu)` vs `mx.default_stream(mx.cpu)`. Most user code does not need to set this; the default targets the GPU. CPU stream is useful when MLX is used as a fast NumPy on the CPU side of a hybrid pipeline.

## Common pitfalls

1. **Treating `mx.eval` as optional.** Forgetting it means the graph never runs and your timings are nonsense.
2. **Expecting CUDA semantics.** No `to(device)`, no `pin_memory`, no `non_blocking=True`. The substrate is different; the API reflects it.
3. **Unnecessary host syncs.** Calling `.tolist()` or `np.array(...)` forces eval and host copy. Stays inside MLX in hot paths.
4. **Forgetting the memory cap.** `mx.metal.set_memory_limit(int(0.85 * total_ram))` keeps headroom for the OS. On a 64GB Mac, leave at least 8GB.
5. **Loading two large models at once.** Each is a real allocation. There is no "VRAM swapping" — if you exceed available RAM, the OS pages you out and tok/s falls off a cliff.

## What you walk away with

- A working install of `mlx-lm`.
- A 4-bit Qwen2.5-7B downloaded and runnable.
- Comfort with `mx.array`, lazy/eval semantics, and dtypes.
- Understanding that MLX is the Apple-native answer to "what should I use to run LLMs on this Mac" — not PyTorch MPS.

## References

- MLX repo: https://github.com/ml-explore/mlx
- MLX docs: https://ml-explore.github.io/mlx/
- mlx-lm: https://github.com/ml-explore/mlx-lm
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm
- mlx-community on HuggingFace: https://huggingface.co/mlx-community
- Distributed MLX RFC: https://github.com/ml-explore/mlx/discussions/distributed
