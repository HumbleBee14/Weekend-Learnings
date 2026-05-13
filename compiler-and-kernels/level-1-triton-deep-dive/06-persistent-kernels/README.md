# 06 — Persistent kernels and CUDA graphs

> Outer reference: [`../README.md`](../README.md). Prerequisite: sub-modules 03 (you've seen the persistent pattern applied to RMSNorm in [`../03-rmsnorm-bandwidth-journey/05_persistent.py`](../03-rmsnorm-bandwidth-journey/05_persistent.py)) and 04 (tiled matmul).

In sub-module 03 the persistent pattern bought you ~10–20% more bandwidth on RMSNorm. That was the small reason to learn it. The large reason is this sub-module: **persistent kernels are how production inference engines stay inside a CUDA graph.** vLLM v1, the IBM Triton paged-attention kernel ([arXiv 2511.11581](https://arxiv.org/abs/2511.11581), [vLLM blog Mar 2026](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)), and the PyTorch grouped-GEMM MoE ([PyTorch blog](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)) all share one design: fixed launch grid, dynamic tile assignment inside the kernel, capture the whole forward into one graph.

By the end of this folder you can retrofit any of your kernels with the persistent pattern and capture it into a CUDA graph that survives shape changes.

Time budget: 2–3 hours. Free Colab T4 is sufficient — the win is most visible on decode shapes (M=1, M=8) which is exactly where T4 launch overhead dominates.

## The motivation in one paragraph

A normal Triton matmul launches `ceil(M/BLOCK_M) * ceil(N/BLOCK_N)` programs. For decode (M=1, N=hidden_dim, K=hidden_dim) on a llama-shaped model the grid is on the order of 32 programs. The kernel itself runs in microseconds. The launch costs ~5–10 µs. At decode you re-launch this kernel maybe 200 times per token (every linear layer, every attention output projection). Launch overhead alone is 1–2 ms per token. CUDA graphs collapse that to one launch — but only if your grid shape doesn't change between calls. Variable batch / variable seqlen breaks that. Persistent kernels fix it: the grid is *always* `(num_SMs,)` regardless of shape, so one captured graph serves every shape your scheduler hands you.

## What you build

| File | What it teaches |
|---|---|
| [`01_persistent_matmul.py`](01_persistent_matmul.py) | Static-persistent matmul. Each SM owns a precomputed range of tiles. Bench vs the non-persistent kernel from sub-module 04. |
| [`02_dynamic_persistent_matmul.py`](02_dynamic_persistent_matmul.py) | Dynamic persistent: tiles claimed via `tl.atomic_add` on a global counter. Bench on ragged work where tiles cost wildly different amounts. |
| [`03_capture_into_cuda_graph.py`](03_capture_into_cuda_graph.py) | Capture the static-persistent kernel into a `torch.cuda.graph`. Measure the launch-overhead saving on decode shapes. |
| [`04_piecewise_graph_demo.py`](04_piecewise_graph_demo.py) | The vLLM-style piecewise pattern: capture the static (persistent) kernels into one graph, leave the variable-shape op outside, replay across shape changes. |
| [`CONCEPTS.md`](CONCEPTS.md) | The deep dive — derivation of the static vs dynamic schedule, what makes a kernel graph-friendly, what piecewise CUDA graphs actually save. |
| [`notes.md`](notes.md) | Where you record your numbers. |

## What to do

1. Read [`CONCEPTS.md`](CONCEPTS.md) before any code. The static/dynamic distinction and the graph-friendliness rules are the load-bearing ideas.
2. Run [`01_persistent_matmul.py`](01_persistent_matmul.py). Note the small (often single-digit %) win on large square shapes; note the bigger (often 1.5–3×) win on decode shapes where the non-persistent grid is small.
3. Run [`02_dynamic_persistent_matmul.py`](02_dynamic_persistent_matmul.py). The synthetic-ragged benchmark is rigged to favor dynamic — it's how split-K and variable-seqlen attention behave. Watch tail latency.
4. Run [`03_capture_into_cuda_graph.py`](03_capture_into_cuda_graph.py). This is where the story pays off: at M=1 the captured-graph version should beat plain eager by 2–5×, almost all of that from launch-overhead elimination.
5. Run [`04_piecewise_graph_demo.py`](04_piecewise_graph_demo.py). Read its comments carefully — the structure mirrors what vLLM does in `vllm/v1/cudagraph_dispatcher.py`.
6. Write three sentences in [`notes.md`](notes.md): one observation per file 01–04.

## Where this goes next

Level 2 of this track is about `torch.compile`. The compiler's whole reason for being is to produce kernels that look like the ones in this folder — fused, persistent, graph-captured. You'll see Inductor emit Triton that uses exactly these idioms. After Level 1 you'll be reading Inductor output; after Level 2 you'll be modifying it.

The capstone fused RMSNorm+RoPE in `_capstone-fused-rmsnorm-rope/` does not require persistence to hit Liger parity (Liger itself isn't persistent for RMSNorm) — but if you take your capstone and put it into a piecewise graph for a decode workload, you've replicated the structural piece that makes vLLM v1 fast on small batches. That's the connection.

## Pitfalls

1. **Persistent doesn't always win.** On large square shapes the non-persistent grid already saturates the hardware; persistence buys you nothing or even loses a percent or two. Persistent wins on (a) small/skewed grids where launch overhead matters or (b) workloads where you need the graph capture. Don't apply it everywhere by reflex.
2. **Atomic counter has a cost.** `tl.atomic_add` on global memory serializes through the L2. For small tile counts (decode), the static schedule is better. For large unbalanced tile counts, dynamic wins. There's a crossover — find it on your hardware in file 02.
3. **CUDA graphs assume fixed input/output pointers and shapes.** If your scheduler hands you a new `query` tensor each iteration with a new address, the captured graph reads from stale memory. You either (a) reuse the same buffer (the vLLM approach — pre-allocate, copy in) or (b) recapture. The first call in file 03 walks through this.
4. **`torch.cuda.graph` captures the launch, not the autotune.** Autotuning launches multiple test kernels — capture happens *after* autotune resolves. File 03 warms up with `do_bench` first to lock in the best config.
5. **Stream semantics matter.** Capture happens on a non-default stream. If you mix default-stream work in, capture fails with cryptic errors. File 03 shows the canonical idiom.

## Resources

- [PyTorch — Accelerating MoEs with a Triton Persistent Cache-Aware Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) — the production reference for this pattern.
- [vLLM Triton Attention Backend Deep Dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) (Mar 2026) — sections on persistent scheduling and piecewise CUDA graphs.
- [Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581) — the ~800-LoC paged-attention kernel; the work-splitting scheme is what file 02 mimics on matmul shapes.
- [PyTorch — Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
- [PyTorch docs — CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
- [Triton tutorial — Persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html).
