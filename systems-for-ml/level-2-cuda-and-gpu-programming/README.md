# Level 2 — CUDA & GPU Programming

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: feeds Project 1 (KV cache work in Level 4) and Project 2 (engine internals)

## Week goal

Stop treating the GPU as a black box. By Friday you should be able to:

- Sketch the GPU execution model on a whiteboard — threads, warps, blocks, SMs, the memory hierarchy — without looking it up.
- Write a working CUDA C++ kernel for vector addition and a tiled matrix multiply.
- Write the same matmul in Triton (Python) and beat your naive CUDA version.
- Read FlashAttention-2 (and at least the abstract of FA3) and trace, in your own words, why it's a memory-bandwidth optimization rather than a compute optimization.
- Know what FlashInfer is and why it sits underneath every modern serving engine.

This is the only week in the curriculum where you write substantial GPU code yourself. After this, the goal is to *read* and *tune* kernels, not write them — but you can't tune what you can't read.

## Where this fits

- **Comes after:** Level 1 (you have a server, you've seen single-GPU latency).
- **Comes before:** Level 3 (you'll profile real GPU code and need the mental model to interpret traces), Level 4 (kernel fusion, FlashAttention conceptually, paged attention internals).
- **Project this feeds:** Indirect — kernel literacy is the prerequisite for the KV cache work in Level 4 and the engine bake-off in Level 5.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | cuda-mental-model | Threads / blocks / warps / SMs — diagram and explain |
| 02 | first-cuda-kernels | Vector add, elementwise ops in CUDA C++ |
| 03 | matrix-multiply | Naive → tiled → shared memory matmul |
| 04 | triton-intro | Same kernels in Triton (Python) |
| 05 | gpu-memory-hierarchy | HBM vs L2 vs SRAM — why FlashAttention exists |
| 06 | flash-attention-walkthrough | Read FA2/FA3, identify each tile/load step |

### 01 — `cuda-mental-model`

**What it is.** The execution hierarchy a GPU actually uses. A kernel launches a *grid* of *blocks*. Each block contains *threads*, grouped into *warps* of 32 (NVIDIA) that execute in lockstep (SIMT). Blocks are scheduled onto *streaming multiprocessors (SMs)*. An H100 has 132 SMs; a consumer 4090 has 128.

**Why it matters.** Every performance question on a GPU comes back to this hierarchy. "Why is my kernel slow?" usually means one of: warp divergence (threads in the same warp taking different branches), low occupancy (not enough warps per SM to hide latency), shared-memory bank conflicts, or memory-bandwidth saturation. You can't reason about any of those without the mental model.

**What to read.**
- NVIDIA's [CUDA Programming Guide §1–§3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — sections 1 through 3 are enough.
- The [CUDA mode lecture series](https://github.com/cuda-mode/lectures) on YouTube — Lectures 1, 3, 4. Free, excellent.

**Output.** A diagram in your notes — handwritten or in `excalidraw` — that shows: launch → grid → blocks → warps → threads, alongside the memory hierarchy (registers, shared, L2, HBM). You will reference this diagram for the rest of the curriculum.

### 02 — `first-cuda-kernels`

**What it is.** Vector addition, elementwise ReLU, elementwise softmax — kernels small enough that you can write them in 30 minutes each. Goal is mechanical fluency with `__global__`, `blockIdx.x`, `threadIdx.x`, `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`.

**Build steps.**
1. Use Google Colab with a T4 (free). Save the notebook to your repo.
2. Write `vector_add.cu`. Launch with `<<<num_blocks, threads_per_block>>>`. Verify against a CPU reference.
3. Write `elementwise_relu.cu`. Time it with `cudaEvent`s, compare to PyTorch's `relu()`. PyTorch will win — that's expected, it's been tuned for years.
4. Write `softmax.cu`. This one is harder because it has reductions. Naive version: each thread does its own row. Better version: one block per row, threads in the block cooperate via shared memory.

**What to measure.** Achieved bandwidth — `(bytes_read + bytes_written) / time`. Compare to your GPU's peak HBM bandwidth (T4: ~320 GB/s, A100: ~2 TB/s, H100: ~3.4 TB/s). If you're at <30% of peak on a memory-bound kernel, you've got room to optimize.

**Common confusion.** "Why is the first launch slow?" CUDA context init + kernel JIT happens lazily. Always do a warmup launch before timing.

### 03 — `matrix-multiply`

**What it is.** Three implementations of the same math, increasing in sophistication:
1. **Naive** — one thread per output element, reads A row + B column from HBM directly. Memory-bandwidth-bound, very slow.
2. **Tiled (shared memory)** — block computes a tile of C cooperatively. Threads load A and B tiles into shared memory once, reuse across multiple output elements. ~10× faster.
3. **Optimized tiled** — register tiling on top of shared memory. Each thread computes a small block of outputs to maximize register reuse. Approaches cuBLAS performance for small/medium matrices.

**Why it matters.** Matmul is the kernel underneath every LLM. Every weight projection, every attention computation. If you've tiled a matmul yourself, the rest of the field — FlashAttention, GEMM kernels, all of CUTLASS — stops being mysterious.

**What to read.** Simon Boehm's ["How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"](https://siboehm.com/articles/22/CUDA-MMM) is the canonical walkthrough. Read it twice. Implement up to kernel 6.

**What to measure.** TFLOPS achieved. T4 peak is ~8 TFLOPS (FP32). Naive will hit <1 TFLOPS; tiled will hit 3–5; cuBLAS will be at 6+. Plot it.

### 04 — `triton-intro`

**What it is.** OpenAI Triton — a Python DSL that compiles to PTX. You write kernel logic in Python with NumPy-like operations on tiles, and Triton handles the low-level CUDA-isms (shared memory allocation, scheduling, vectorization).

**Why it matters.** Triton is what serving engines actually use for custom kernels. vLLM, SGLang, and Unsloth all have Triton kernels. NVIDIA's TensorRT-LLM increasingly does too. The reason: writing CUDA C++ for every model variant is unsustainable, but cuBLAS isn't always the right fit. Triton is the sweet spot.

**Build steps.**
1. `pip install triton` (Linux + NVIDIA only as of 2026; Mac/AMD support is partial).
2. Write the same matmul as Step 03 in Triton. ~50 lines instead of ~200.
3. Compare performance. On T4 your tuned Triton matmul should beat your hand-tuned CUDA tiled version, because Triton's auto-tuner picks better block sizes than you guessed.
4. Write a fused `softmax(x) * y` kernel. This is where Triton starts to shine — fusion is what eliminates HBM round-trips.

**What to read.** The [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), specifically `01-vector-add`, `03-matrix-multiplication`, and `06-fused-attention`.

**Insight to carry.** Once you've written a fused kernel in Triton, you understand 80% of why FlashAttention is fast: it fuses the attention computation so QK^T and softmax(QK^T)V never round-trip to HBM as full matrices.

### 05 — `gpu-memory-hierarchy`

**What it is.** GPUs have a memory hierarchy with absurd bandwidth deltas:

| Level | Size (H100) | Bandwidth |
|-------|-------------|-----------|
| Registers (per thread) | 256 × 4B | unmeasured (effectively free) |
| Shared memory / L1 (per SM) | 228 KB | ~20 TB/s |
| L2 cache (chip-wide) | 50 MB | ~5 TB/s |
| HBM3 (chip-wide) | 80 GB | ~3.4 TB/s |

The bandwidth gap between SRAM (shared memory) and HBM is ~6×. That ratio is the entire reason FlashAttention exists.

**Why it matters.** For LLM inference and training, the bottleneck is almost never compute — it's memory bandwidth. A naive attention computation generates an N×N intermediate matrix, writes it to HBM, then reads it back for the softmax. FlashAttention computes attention tile-by-tile so that intermediate stays in SRAM and never round-trips. Same FLOPs, much less HBM traffic, much faster.

**What to read.** The original [FlashAttention paper, Section 3](https://arxiv.org/abs/2205.14135) — read just Section 3 ("FlashAttention: Algorithm, Analysis, and Extensions") this week. Skip the proofs.

**Output.** Three sentences in your notes answering: *Why does naive attention have O(N²) memory but FlashAttention has O(N)?* If you can answer that crisply you've got it.

### 06 — `flash-attention-walkthrough`

**What it is.** Reading the FA2 and FA3 papers (or excellent secondary sources) until you can trace, step by step, what a single tile of QK^T → softmax → V looks like in registers and shared memory.

**FlashAttention versions in 2026.**
- **FA2** — the standard. Works on Ampere (A100) and consumer Ada (4090). Default kernel in PyTorch SDPA on those GPUs.
- **FA3** — Hopper-only (H100/H200). 1.5–2× faster than FA2 by exploiting Hopper's TMA (tensor memory accelerator) and warp specialization. ~75% of theoretical peak FLOPS, with FP8 support.
- **FlashInfer** — not a competitor to FA, but the *kernel layer underneath* vLLM, SGLang, and TRT-LLM. It dispatches to FA2/FA3/cuDNN/CUTLASS based on the workload, and beats raw FA3 on batched decode (uneven sequence lengths) by up to 3× with a load-balanced scheduler. You won't write FlashInfer kernels, but you should know it exists — when someone says "vLLM's attention kernel," they probably mean FlashInfer.

**What to read.**
- Tri Dao's [FA3 PyTorch blog post](https://pytorch.org/blog/flashattention-3/) — short, well-illustrated.
- The [FlashInfer GitHub README](https://github.com/flashinfer-ai/flashinfer) — at minimum the architecture diagram.

**Output.** A 200-word writeup in your notes: *"How does FlashAttention compute attention without materializing the N×N matrix?"* with the online softmax trick spelled out. This will be hard to write the first time. That's the point.

## Project work this week

There's no Project 1/2/3/4 deliverable specifically due this week — Level 2 is foundation-building. But two things from this week feed forward:

1. **Your tiled matmul code** (Triton version) — keep it. In Level 4 you'll reference its memory-access pattern when reasoning about KV cache layout.
2. **Your FlashAttention writeup** — keep it. In Level 5 (engine bake-off), when you compare engines on long-context workloads, you'll connect the throughput differences back to which FA variant each engine uses.

## Definition of done

- [ ] You can sketch the GPU execution hierarchy from memory.
- [ ] You have a working CUDA matmul that hits >50% of cuBLAS throughput on T4.
- [ ] You have a working Triton matmul that beats your hand-tuned CUDA.
- [ ] You have a 200-word writeup of FlashAttention's tile-and-online-softmax trick in your own words.
- [ ] You can name FA2 vs FA3 vs FlashInfer and explain when each is in play.

## Resources (canonical only)

- **CUDA Programming Guide** — [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). Skim §1–§3.
- **CUDA mode lectures** — [github.com/cuda-mode/lectures](https://github.com/cuda-mode/lectures). Lectures 1, 3, 4 this week.
- **Simon Boehm — CUDA matmul** — [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM). Required reading.
- **Triton tutorials** — [triton-lang.org tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html). Tutorials 01, 03, 06.
- **FlashAttention paper** — [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135). Section 3 only.
- **FA3 blog** — [pytorch.org/blog/flashattention-3](https://pytorch.org/blog/flashattention-3/).
- **FlashInfer** — [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).

## Common pitfalls

1. **Going too deep into CUDA C++ this week.** Aim for *literacy* here — enough to read kernels, reason about them, and tune them. Deep CUDA mastery is its own subject; if it pulls you in, it's worth pursuing as a separate, longer track later. Two weeks inside a 9-week curriculum is the right dosage for now.
2. **Skipping Triton because "I already wrote CUDA."** Triton is what production engines use. In 2026 it's part of the standard kit alongside CUDA — please don't skip it.
3. **Believing benchmarks without warmup.** First kernel launch includes JIT + context init. Always warmup.
4. **Comparing FP32 vs FP16 vs FP8 numbers as if they're equivalent.** They're not. State the precision when you report TFLOPS. cuBLAS at FP16 is ~2× FP32 on the same hardware.
5. **Skipping FlashAttention because the paper looks scary.** Section 3 is 4 pages and accessible. Skip the proofs. The intuition is what you need.

## What you'll be able to do after this week

> Implement matrix multiplication kernels in CUDA C++ (naive → tiled → register-tiled) and in Triton, hitting a meaningful fraction of cuBLAS throughput. Read FlashAttention-2/3 and explain the SRAM-vs-HBM bandwidth tradeoff that motivates tiled attention.

You won't write FlashAttention this week — at this stage, the goal is to *understand* it. Aim for kernel literacy. If kernel mastery interests you, it's a worthwhile separate journey on its own; Level 9 points at where to go next.
