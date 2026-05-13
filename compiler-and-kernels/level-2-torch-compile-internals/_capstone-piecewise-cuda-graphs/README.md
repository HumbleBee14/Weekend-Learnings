# Capstone — Piecewise CUDA graphs on a LLaMA decoder block

This is the level. You take one LLaMA-shaped decoder block, walk it through five compilation strategies, and end with a working piecewise CUDA graph wrapper that mirrors vLLM v1's design. The wrapper code is parameterized — you can lift it into another project.

By the time you finish, you should be able to read [`vllm/compilation/cuda_graph.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_graph.py) and recognize most of what's happening.

## Why piecewise

Two facts in tension:

1. **Most of a decoder block has fixed shape** (per token) — the linears, the norms, the MLP. CUDA-graphing this region removes 50–200 µs of kernel launch overhead per call. For decode (batch×1) this is most of the latency.
2. **Attention's shape changes every step** — KV cache grows. Capturing attention into a CUDA graph means re-capturing every step, which is *more* expensive than the launch overhead saved.

The vLLM solution: wrap attention as a custom op so Dynamo treats it as one opaque node, then let Inductor partition the FX graph at that node and CUDA-graph each non-attention piece independently. Attention runs eagerly in between.

The pattern, drawn:

```
  ┌──────────────────────────────────┐
  │  CUDA Graph 1                    │
  │  RMSNorm → QKV proj → RoPE       │
  └────────────────┬─────────────────┘
                   ▼
         [attention — eager]
                   ▼
  ┌──────────────────────────────────┐
  │  CUDA Graph 2                    │
  │  O proj → residual → RMSNorm     │
  │  → MLP → residual                │
  └──────────────────────────────────┘
```

Each CUDA graph is captured once per (batch, seqlen) bucket. Attention runs eager and adapts to the changing KV cache shape every step.

## Hardware

T4 minimum. Decode-shape numbers are most striking on smaller GPUs (T4) because launch overhead is a larger fraction of total work. Big H100 numbers will show smaller relative wins because the GPU is faster at everything.

## What's in this folder

| File | What it is |
|---|---|
| [`llama_block.py`](llama_block.py) | One LLaMA decoder block, ~150M params at hidden=4096. Eager-mode reference. |
| [`piecewise_wrapper.py`](piecewise_wrapper.py) | The reusable piecewise CUDA graph wrapper. Lift this into your project. |
| [`benchmark.py`](benchmark.py) | Five variants timed against each other; produces the capstone table. |
| [`audit.py`](audit.py) | Runs the block through `fullgraph=True` + depyf to surface graph breaks. |
| [`notes.md`](notes.md) | Template for your write-up. |

## What to do, in order

1. **Run the audit.** `python audit.py`. It will list every graph break in the block. Fix them in [`llama_block.py`](llama_block.py) until `fullgraph=True` passes. (Hint: at least one `.shape` access inside SDPA's call site can be problematic; HuggingFace-style cache plumbing has been deliberately *not* included here to keep the focus tight.)

2. **Run the benchmark.** `python benchmark.py`. It runs five variants:
   - Eager
   - `torch.compile(mode="default")` with breaks still in place
   - `torch.compile(mode="default")` with breaks fixed, `fullgraph=True`
   - `torch.compile(mode="reduce-overhead")` — full CUDA graph
   - **Your piecewise wrapper.**

3. **Fill in your numbers** in [`notes.md`](notes.md). Compare prefill (B=1, S=128) and decode (B=1, S=1, KV cache prefilled to length 128). The piecewise win shows up most in prefill, where the attention shape is changing.

## The piecewise wrapper, conceptually

You implement (in [`piecewise_wrapper.py`](piecewise_wrapper.py)):

1. **Wrap attention as a custom op.** This is the trick that makes the graph one piece from Dynamo's view:
   ```python
   @torch.library.custom_op("capstone::attn", mutates_args=("kv_cache",))
   def attn(q, k, v, kv_cache, ...):
       # eager SDPA call, can vary shape freely
       ...
   ```

2. **Compile the whole block with `torch.compile`.** Inductor sees: a chain of ops, one of which is the custom op. With Inductor graph partition enabled (`torch._inductor.config.graph_partition = True` in 2.8+), Inductor splits the FX graph at the custom op.

3. **Wrap each partition in a CUDA graph at capture time.** The capture happens lazily on the first call with new shapes — you maintain a dict keyed by `(batch, seqlen)` of captured graphs and replay them.

4. **Dispatch at call time.** For each forward call: look up the right captured graph for this shape, copy inputs into the graph's input buffers, replay, copy outputs out. Attention runs eagerly between the graphs.

You don't have to write the CUDA graph machinery from scratch — `torch.cuda.CUDAGraph` + `torch.cuda.graph()` is the API. The work is in **the dispatcher**: choosing the right graph for the current shape, and pre-allocating the input/output buffers correctly.

## Reference (read these while you build)

- [vLLM CUDA Graphs design doc](https://docs.vllm.ai/en/stable/design/cuda_graphs/) — your design target.
- [`vllm/compilation/cuda_graph.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_graph.py) — the reference implementation. Read after you've written your own draft. Comparing is more useful than copying.
- [PyTorch CUDA Graphs docs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
- [Aug 2025 vLLM blog](https://blog.vllm.ai/2025/08/20/torch-compile.html) — the conceptual walkthrough.

## Definition of done

- [ ] `audit.py` passes with `fullgraph=True` and zero graph breaks.
- [ ] `benchmark.py` produces the five-row table; numbers fill in correctly.
- [ ] Your piecewise wrapper matches or beats `reduce-overhead` on prefill (the variable-shape case).
- [ ] Your piecewise wrapper code is parameterized over the block module — i.e. you could pass any `nn.Module` whose forward has one `attn(...)` call and the wrapper would work.
- [ ] You wrote a one-page reflection in [`notes.md`](notes.md) covering: where the win came from, where you fell short of vLLM, what you'd change to handle batch > 1.

## What this earns you

You have re-implemented the load-bearing optimization in vLLM v1. You can read the rest of `vllm/compilation/` and follow it. You can apply the pattern to any model with a "one shape-variable op surrounded by fixed-shape compute" structure — which is most modern inference workloads (Mamba, MoE expert routing, image generation with variable resolutions, etc.).
