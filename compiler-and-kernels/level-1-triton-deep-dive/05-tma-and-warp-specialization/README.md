# 05 — TMA and warp specialization

> Previous: [`04-tiled-matmul-and-autotune`](../04-tiled-matmul-and-autotune/) — you wrote a tiled GEMM with [`tl.make_tensor_descriptor`](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html) and autotuned it.
>
> Next: [`06-persistent-kernels`](../06-persistent-kernels/) — you fix the grid to `num_SMs` and own the scheduler.

You have a tiled GEMM that hits ~70% of cuBLAS on H100. The compute is hot, the descriptors are correct, autotune found a sane config. But the tensor cores still spend a meaningful fraction of every iteration waiting for the next tile to land in shared memory. That gap is what this sub-module closes.

You add **one line** — `warp_specialize=True` in your `tl.range` loop — and the compiler splits the loop body into producer warps that issue TMA loads and consumer warps that run MMAs, running them in parallel. This is the pattern that bought FlashAttention-3 its 1.5–2× over FA2, that lets vLLM's ~800-line Triton attention kernel match FA3, and that is now the default shape of every fast GEMM and attention kernel on Hopper and Blackwell.

Time budget: 2–3 hours if you have Hopper or Blackwell. 1 hour if you don't (you read the trace in [`04_no_hopper_fallback.md`](04_no_hopper_fallback.md) instead of running 02 and 03).

## Hardware honesty

Read this before you commit to running the scripts.

- **Hopper (H100, H200) or Blackwell (B200, GB200) gets you the real lesson.** Producer/consumer with TMA is a Hopper feature. You will see the speedup on these chips.
- **Ada (RTX 4090, L40S) compiles but barely moves the needle.** No TMA hardware; the descriptor path lowers to ordinary `cp.async` and warp specialization helps only marginally. Expect 0–10%.
- **Ampere (A100, RTX 3090) and Turing (T4) compile but warp specialization either no-ops or falls back.** The kernels still run and produce correct results; the speedup you measure is small or zero. This is fine — the code paths are educational even when the speedup isn't there.
- **AMD MI300/MI325** runs the same source. The Triton 3.7 release made wave specialization with TDM (AMD's TMA analog) the default on these chips. You should see speedups in the same ballpark as H100.

Every script prints a one-line diagnostic at start telling you what your GPU supports. If it says `warp_specialize: ignored on this arch`, that is the expected output on T4 / A100 — keep going for the structural lesson, ignore the absolute numbers.

If you have no Hopper / Blackwell / MI300 at all, skip 02 and 03 and read [`04_no_hopper_fallback.md`](04_no_hopper_fallback.md). You will not be blocked in 06 or the capstone.

## What you build

| File | What it teaches | Hardware |
|---|---|---|
| [`01_tma_matmul.py`](01_tma_matmul.py) | The same tiled GEMM from 04 but stripped to its TMA-only form — no warp spec yet. Bench vs cuBLAS. | any CUDA |
| [`02_warp_specialized_matmul.py`](02_warp_specialized_matmul.py) | Add `warp_specialize=True` to the K-loop. Tune `num_consumer_groups` and `num_buffers_warp_spec`. Bench vs 01. | Hopper+ for the speedup |
| [`03_warp_specialized_attention.py`](03_warp_specialized_attention.py) | A minimal FA2-style forward attention with warp specialization on the inner K loop. Bench vs `torch.scaled_dot_product_attention`. | Hopper+ for the speedup |
| [`04_no_hopper_fallback.md`](04_no_hopper_fallback.md) | An annotated proton trace and an H100 SOL table for learners without the hardware. | none |
| [`notes.md`](notes.md) | Observations template. | none |
| [`CONCEPTS.md`](CONCEPTS.md) | The deep treatment — what TMA actually is, the producer/consumer mental model, the Tawa `aref` abstraction, what changed on Blackwell, when warp spec doesn't help. | none |

## What to do

1. Read [`CONCEPTS.md`](CONCEPTS.md). Don't skip — the runnable scripts are short and only make sense once you can picture the producer warps issuing TMA copies in parallel with the consumer warps running MMAs.
2. Run [`01_tma_matmul.py`](01_tma_matmul.py). Record GB/s and TFLOPS. Confirm correctness against `torch.matmul`.
3. Run [`02_warp_specialized_matmul.py`](02_warp_specialized_matmul.py). Same shape, same dtype. Record the delta. On H100 expect 1.2–1.5×. On non-Hopper expect roughly flat — the diagnostic at the top of the script tells you which case you're in.
4. Run [`03_warp_specialized_attention.py`](03_warp_specialized_attention.py). Compare against `torch.nn.functional.scaled_dot_product_attention` (which dispatches to FA3 on Hopper, FA2 elsewhere). Don't expect to beat FA3 — you won't, because FA3 is hand-tuned in CuTe. The lesson is *how close* you get with ~150 lines of Triton.
5. Profile your fastest matmul with [`triton.proton`](https://triton-lang.org/main/profiling/proton.html) and look at SM occupancy and tensor-core utilization. Confirm: tensor-core active cycles went up, HBM-stall cycles went down. That is the speedup, in two numbers.
6. Write three sentences in `notes.md` describing — in your own words — what the producer and consumer warps are doing on each iteration of the K-loop. If you can't, re-read `CONCEPTS.md` section "The async pipeline mental model" and try again.

The discipline: **don't measure 02 before 01 runs correctly.** A warp-specialized matmul that produces wrong numbers is meaningless no matter how fast. Always sanity-check against `torch.matmul` before believing the timer.

## What you should see

On H100 at `M=N=K=4096`, fp16, TMA-only vs warp-specialized:

| Kernel | TFLOPS | % of H100 fp16 peak (989 TFLOPS) |
|---|---|---|
| `01_tma_matmul.py` | ~600 | ~61% |
| `02_warp_specialized_matmul.py` | ~780 | ~79% |
| cuBLAS (`torch.matmul`) | ~820 | ~83% |

Numbers are rough — your H100 will land within ±10%. On B200 the absolute TFLOPS roughly double (Blackwell's 5th-gen tensor cores), and the warp-spec gap over non-warp-spec widens because the producer/consumer overlap is even more critical for the deeper pipeline. On T4 / A100 the two kernels run within a few percent of each other; that is the expected (boring) result and tells you correctly that warp specialization is a Hopper-and-up feature.

For attention at batch=2, heads=16, seq=4096, headdim=64, fp16:

| Kernel | TFLOPS | Notes |
|---|---|---|
| `torch.nn.functional.scaled_dot_product_attention` | ~580 | Dispatches to FA3 on H100 |
| `03_warp_specialized_attention.py` (yours) | ~450–520 | ~80–90% of FA3 in ~150 lines of Triton |

## Where this goes next

[`06-persistent-kernels`](../06-persistent-kernels/) takes this kernel and fixes the grid to `num_SMs`, so the hardware never re-schedules and the kernel captures into a CUDA graph. That gets you the last decode-latency win and is the pattern vLLM v1 uses.

After Level 1 you should be able to open [Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581) (Oct 2025) and the [vLLM Triton attention deep dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) and follow them. Both are built directly on the pattern you wrote in this sub-module.

## Resources

- [Tawa paper — arXiv 2510.14719](https://arxiv.org/abs/2510.14719) — the formal description of Triton's warp specialization and the `aref` IR abstraction.
- [Triton PR #6288](https://github.com/triton-lang/triton/pull/6288) — the upstream warp specialization implementation.
- [PyTorch: Warp Specialization in Triton — Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) (Jan 2026) — the Meta team's design overview, including the producer/consumer channels and the role of `aref`.
- [Ian Barber: How does Triton do Warp Spec?](https://www.ianbarber.com/blog/2025/05/15/triton-warp-spec/) (May 2025) — a readable walkthrough of how `warp_specialize=True` actually compiles, with PTX snippets.
- [Tri Dao: FlashAttention-3](https://tridao.me/blog/2024/flash3/) — the original producer/consumer attention design that warp specialization in Triton was reverse-engineered from.
- [NVIDIA: OpenAI Triton on Blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/) — what Blackwell adds; MXFP8/MXFP4, 5th-gen tensor cores.
- [Modal: We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4) — why FA4 lives in CuTe-DSL and what that implies about Triton's ceiling on Blackwell.
- [vLLM Triton Attention Backend Deep Dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) (Mar 2026) — the production kernel reading list pays off here.
