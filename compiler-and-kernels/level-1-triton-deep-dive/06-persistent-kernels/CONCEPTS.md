# Persistent kernels, CUDA graphs, and the connection between them

You met the persistent pattern in [`../03-rmsnorm-bandwidth-journey/05_persistent.py`](../03-rmsnorm-bandwidth-journey/05_persistent.py) as a bandwidth trick — keep the SM warm, reuse the weight from L2. That framing is correct but incomplete. The full reason the pattern exists, the reason vLLM v1's [paged-attention kernel](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) is persistent and the PyTorch [grouped-GEMM MoE](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) is persistent, is this: **persistence makes the launch grid independent of the input shape, which is the precondition for capturing the kernel into a CUDA graph that survives shape changes.**

This document derives that claim from first principles.

## The non-persistent baseline and what it costs

For a matmul of shape `(M, N) = A @ B` with tile `(BLOCK_M, BLOCK_N)`, the standard launch is

```python
grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
matmul_kernel[grid](...)
```

Two things move when `M` changes:

1. The number of programs in the grid.
2. Which program handles which output tile.

The hardware scheduler picks up programs and dispatches them onto SMs in *waves* — one wave fills every SM with one program. If you have 132 SMs (H100) and 256 programs, you get two waves; if 64 programs, you get one half-occupied wave. The math is exact: every program lives entirely on one SM, runs to completion, and the SM picks up the next queued program.

Three costs accrue:

- **Launch overhead.** ~5–10 µs per kernel launch on a modern NVIDIA driver (Linux, recent CUDA). Compile and dispatch the launch packet, validate the grid, signal the device. Independent of grid size. Measured on T4 by [PyTorch's CUDA graphs blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/): ~7 µs minimum per launch on hot path.
- **Per-wave scheduling cost.** Each wave transition the SM reads its work queue. Small but non-zero, and it grows linearly with wave count.
- **Cold L2 every wave.** When a new program starts on an SM, the data it needs may have been evicted by the previous program's footprint. The first wave's L2 hit rate is the highest; subsequent waves trend down.

For *training* (large batches, large matmuls, hundreds of waves per launch) the per-launch overhead is negligible — kernels run for milliseconds. For *decode* (M=1, M=8, dozens of programs, sub-millisecond kernels) launch overhead can be 50% of the wall-clock time. This is exactly the regime LLM inference runs in.

## The persistent pattern: schedule yourself

A persistent kernel inverts the relationship. You launch *exactly* `num_SMs` programs, regardless of the work. Each program is sticky — it sits on one SM for the kernel's whole lifetime. Inside, the kernel loops over tiles, picking each one off either a precomputed schedule or a global counter.

The grid becomes `(num_SMs,)`, period. For the same matmul of shape `(M, N, K)`:

```python
total_tiles = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)
num_sms = torch.cuda.get_device_properties(...).multi_processor_count
grid = (num_sms,)

@triton.jit
def kernel(...):
    pid = tl.program_id(0)
    # Static: this pid owns tiles [start_tile, end_tile)
    tiles_per_program = cdiv(total_tiles, num_sms)
    start_tile = pid * tiles_per_program
    end_tile = min(start_tile + tiles_per_program, total_tiles)
    for tile in range(start_tile, end_tile):
        # decode tile -> (m_block, n_block), do the work
        ...
```

The grid doesn't depend on `M`, `N`, or `K`. It depends only on the GPU. The kernel function itself reads `total_tiles` as a runtime argument and loops accordingly.

That's the static-persistent flavor. It works when tile costs are roughly equal, because tiles are dealt out evenly. For our matmul of equal-sized tiles that's fine. For workloads where one tile takes 10× another (think variable-seqlen attention, or grouped-GEMM MoE where each expert has different token counts), one program finishes early and sits idle while another grinds away. The fix:

## Dynamic persistent: atomic claim

Instead of precomputing each program's range, every program races on a global counter:

```python
@triton.jit
def kernel(..., tile_counter):
    while True:
        tile_id = tl.atomic_add(tile_counter, 1)
        if tile_id >= total_tiles:
            return
        # process tile_id
```

Whoever finishes the previous tile claims the next one. Slower tiles slow down their owning SM but don't slow down others. This is exactly the work-stealing pattern used in the [vLLM Triton paged-attention kernel](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) for variable-length decode batches, and it's what the [arXiv anatomy paper](https://arxiv.org/abs/2511.11581) calls "dynamic work splitting." The atomic itself isn't free — it serializes through L2 — but for ragged work the load-balance win dwarfs the atomic cost.

The crossover rule of thumb: **if `coefficient_of_variation(tile_cost) > 0.3`, use dynamic; otherwise static.** You'll measure this yourself in `02_dynamic_persistent_matmul.py`. For square GEMMs, static. For attention with variable seqlen, dynamic. For grouped-GEMM with skewed expert loads, dynamic.

## What CUDA graphs actually are

A CUDA graph is a pre-recorded sequence of CUDA operations — kernel launches, memcopies, event waits — that you record once and replay many times. The recording captures the *operations*, including their parameters (kernel function pointer, grid dim, block dim, kernel arguments). Replay submits the whole sequence to the GPU as one work unit. The CPU does no per-kernel work during replay; the runtime hands the device an instruction stream.

The wins:

- **One driver round-trip instead of N.** A forward pass with 1000 kernel launches replays as one. Launch overhead drops from `N × 7µs` to ~7µs.
- **No Python on the hot path.** Replay happens entirely device-side. Your Python interpreter is free to prepare the next iteration.
- **Deterministic launch order.** The runtime knows the dependency graph exactly. Some optimizations are only safe inside a graph.

The catches:

- **Captured arguments are baked in.** When you record a kernel launch with `grid=(M_blocks,)`, that exact `M_blocks` value is in the graph. Replay launches with the same grid. Change `M`, you need a new grid → new graph → recapture.
- **Captured pointers are baked in.** If you record `kernel[grid](x.data_ptr(), ...)`, that exact memory address is captured. If you allocate a new `x` for the next iteration, the graph reads from the *old* address. You have to reuse buffers or use `torch.cuda.make_graphed_callables` which handles input redirection.
- **No data-dependent control flow.** A graph is a fixed DAG. If your forward has `if x.sum() > 0: A else: B`, you can't capture it. You'd need separate graphs or graph-break around the branch.
- **Capture happens on a non-default stream.** Default-stream operations can't participate. The canonical idiom uses a side stream and synchronizes around capture.

## Why persistent + graph is the right pair

Put the two stories together. You have a decode workload. Each forward pass runs 200 kernels. Sequence lengths vary across batches — sometimes 1, sometimes 8, sometimes 32 active sequences with different lengths each.

Without persistence: every shape change demands a new graph. You'd have a graph for each `(num_active_seqs, max_seqlen)` combo. The graph cache explodes; compile time dominates; you give up.

With persistence: the grid is always `(num_SMs,)`. The kernel sees the shape via runtime tensor arguments. The captured graph launches `(num_SMs,)` programs every time, and inside the kernel they read whatever shape is in the input tensors *now*. One graph serves every shape.

This is the design of [vLLM v1's piecewise CUDA graph dispatcher](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html). Every kernel in the forward is persistent. The whole forward (or all the static parts of it) captures into one graph. At inference time, the scheduler picks a batch, copies inputs into the pre-allocated capture buffers, and replays. The vLLM team measured 5–15% end-to-end throughput improvement from this on Hopper, more on smaller GPUs where launch overhead is a bigger fraction.

## Piecewise: what stays outside the graph

Not every operation is graphable. The major escape hatches in vLLM-style inference:

- **Sampling.** Top-k / top-p sampling has data-dependent control flow.
- **KV-cache allocation.** The block-manager assigns physical KV blocks based on which sequences advanced — that runs on CPU.
- **Variable shapes that don't fit the persistent contract.** Some prefill-only kernels.

So vLLM does *piecewise* graph capture: take the forward, split it at the non-graphable points, capture each contiguous run of graphable kernels into one graph, run the non-graphable bits eagerly between graphs. Looks like:

```
[graph A: embed + N transformer blocks (persistent)] → [eager: sampling] → [graph B: next-iter prep]
```

`04_piecewise_graph_demo.py` builds a minimal version of exactly this structure with two persistent matmuls and an eager `argmax` between them.

## What makes a kernel graph-friendly: a checklist

When you write a kernel intending to capture it:

1. **Fixed launch grid.** Either constant or a function of the device (`num_SMs`), not of the inputs. Persistence is how you get there.
2. **No CPU-side decisions inside the kernel call.** No `if x.shape[0] > 100:` deciding the kernel variant; pick the variant outside.
3. **All shapes flow through tensor arguments.** Pass `M`, `N`, `K` as scalar tensor args if they vary; the kernel reads them at runtime. Don't bake them in via constexpr unless they're truly compile-time.
4. **No memory allocation inside the captured region.** All output tensors pre-allocated; you pass pointers in. `torch.empty` during capture is a common foot-gun.
5. **No host-device sync.** No `.item()`, no `.cpu()`, no `print`. Each forces a sync that breaks capture.
6. **Autotune resolved before capture.** Autotune runs benchmark launches, which interfere with capture. Always warm up first so the best config is selected and cached; then capture.

This checklist is the literal reason persistence wins for inference: it solves item 1, which is the structural one. The rest are coding discipline.

## What you actually measure in this folder

For each file, here's the metric that matters and what to expect on a T4 with the included shapes:

| File | Metric | T4 expectation |
|---|---|---|
| 01 | Persistent vs non-persistent matmul, decode shape (M=1) | Persistent 1.3–2× faster (launch-overhead amortization across few programs) |
| 01 | Same, large square (M=4096) | Roughly tied, within 5% |
| 02 | Dynamic vs static persistent on ragged-K work | Dynamic 1.2–1.5× faster on tail latency (P95) |
| 03 | Graph-captured persistent vs eager persistent at M=1 | 2–5× faster wall-clock |
| 03 | Same at M=4096 | < 10% gain — kernel time dominates launch time |
| 04 | Piecewise graph vs all-eager on a 2-matmul + argmax loop | 1.5–3× depending on shapes |

If your numbers are wildly off in either direction, suspect (a) you measured the first call (autotune + JIT), or (b) the graph never captured the kernel you thought (check `torch.cuda.is_current_stream_capturing` inside the capture region — it should be `True`).

## Generalizable template

After this sub-module the pattern lives in your hands. Whenever you write a new kernel you intend to ship in an inference engine:

1. Write the obvious non-persistent version first. Get it correct.
2. Identify the longest-running call path. If it's launched many times per token with small per-call work, persistence will help.
3. Static-persistent first. If tile costs vary significantly, switch to dynamic.
4. Validate it captures into a CUDA graph cleanly — `torch.cuda.graph` context, replay, compare outputs.
5. Wire it into a piecewise capture if your engine already uses one (vLLM, SGLang, TensorRT-LLM via their Triton paths).

That's the path from "I wrote a kernel" to "the kernel works in a production inference engine." Level 2 of this track picks up here, where `torch.compile`'s Inductor learns to emit exactly these idioms automatically.
