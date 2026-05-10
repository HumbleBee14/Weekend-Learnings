# 08 — AI-Assisted Kernels

The frontier question for this level: now that LLMs can write code, can they write *fast* GPU kernels? The 2025–2026 answer is "sometimes, for narrow shapes, with the right harness, and you still need a human compiler engineer to set up the problem." That's a meaningful shift from 2023, but it's not "kernel writing is solved."

This topic surveys what works, what doesn't, and what the realistic trajectory looks like.

## What "AI-assisted kernel" actually means

Three distinct things get bundled under this label:

1. **LLM-generated source kernels** — give a model a problem ("write a Triton matmul for shape M=N=K=4096 on H100, tune block sizes") and have it produce a `.py` or `.cu` file. The model doesn't run code; another tool benchmarks the output.
2. **LLM-in-the-loop autotuning** — the model proposes configurations, a runner benchmarks them, the model proposes refinements. Closer to evolutionary search guided by an LLM than pure code generation.
3. **LLM-as-compiler-pass** — a small specialized model takes an IR fragment and rewrites it. Used inside experimental compiler stacks, not yet mainstream.

The benchmark / paper landscape mostly addresses (1) and (2). Production usage is mostly (2) — humans still write the scaffolding; the LLM proposes the choice of block sizes, swizzle pattern, or fusion shape inside that scaffolding.

## KernelBench — the eval

KernelBench is the closest thing to a standard benchmark for LLM kernel generation. Released by Stanford / NVIDIA / Princeton (2024–2025), updated through 2025.

- Repo: https://github.com/ScalingIntelligence/KernelBench
- Paper: https://arxiv.org/abs/2502.10517

What it measures: given a PyTorch reference op (matmul, softmax, layernorm, attention variants, etc.), can a model produce a CUDA / Triton kernel that (a) compiles, (b) is numerically correct, (c) is faster than the PyTorch eager baseline?

Results as of late 2025:
- Top models (GPT-class, Claude-class, Gemini-class) write **correct** kernels for ~30–60% of the medium-difficulty problems on first attempt.
- Of the correct kernels, only a fraction are **faster** than baseline; many are correct-but-slow.
- For "speed of light" comparisons against cuBLAS / CUTLASS, LLM kernels almost never win — yet.

The honest read: LLMs are useful as kernel *drafters*. The compile-and-benchmark loop catches their mistakes. Treat the LLM as a starting point, not a final author.

## Sakana AI's CUDA agent (early 2025)

Sakana AI demonstrated an iterative agent that wrote a custom matmul kernel competitive with hand-tuned baselines. The trick wasn't the model alone — it was the harness: many proposals, automatic benchmarking, error-driven refinement, evolutionary mixing of partial solutions.

Two takeaways:

- **The harness is the hard part.** The agent lives or dies by how fast it can compile-and-bench, how accurately it can tell apart "wrong" from "slow," and how cleverly it explores the configuration space. The LLM is one component.
- **The wins were on standard problem shapes.** Novel attention variants and unusual hardware paths (e.g., new Hopper TMA patterns) are still beyond the reach of the agent without human guidance.

Initial blog post (Feb 2025): https://sakana.ai/ai-cuda-engineer/. Note: the post drew correctness criticism (some submitted kernels exploited eval-time shortcuts); read the follow-ups too. The methodology lessons are real even where individual numbers were softened.

## The state in 2026

Where AI-assisted kernel work is genuinely useful:

- **Autotuning sweeps.** An LLM proposing block sizes, num_warps, num_stages for a Triton kernel on a new shape is faster than exhaustive search and roughly as good.
- **Boilerplate generation.** A first-draft Triton kernel for a vanilla op (elementwise, reduction, simple matmul) saves an hour over starting from scratch. The human still tunes.
- **Compiler-pass synthesis.** Research on having an LLM propose new optimization passes (or rewrite rules) for MLIR is showing modest wins. Not in production stacks.

Where it doesn't work yet:

- **Speed-of-light kernels.** FlashAttention-3-class hand-tuned CUTLASS kernels are still beyond LLM-only generation. The reasoning chains required are too long, the feedback signal (perf counter delta from one rewrite) too sparse for current models.
- **Novel hardware targets.** A new accelerator with no public kernels in the model's training data gets nothing useful from an LLM. The vendor's compiler engineers still do all the bring-up.
- **Numerics-sensitive kernels.** Quantized GEMMs, FP8/FP4 epilogues, low-precision attention — the failure modes are subtle (silent precision loss, NaN propagation) and LLMs frequently miss them.

The trajectory: with each model generation, the boundary moves out. What needed a human in 2024 (basic Triton matmul) is solid LLM territory in 2026. What needs a human in 2026 (FA3-class warp-specialized attention) is plausibly LLM territory by 2028 — but the "exploration tax" from compile-and-bench loops is what gates the speed of progress, not raw LLM capability.

## Tools and projects to know

- **KernelBench** — https://github.com/ScalingIntelligence/KernelBench. The eval. If you want to try generating kernels yourself, start here.
- **AutoKernel / AI-CUDA-Engineer style work** — Sakana AI agents. https://sakana.ai/.
- **Triton autotuner with LLM proposals** — multiple research projects through 2025; mostly papers, no single canonical tool yet.
- **OpenAI Triton's autotuner** — non-LLM, exhaustive grid search. The baseline an LLM-driven approach must beat.
- **NVIDIA Inductor work** — internal NVIDIA work on LLM-assisted Triton tuning (presented at PyTorch Conference 2024 and 2025). Limited public artifacts.
- **Anthropic's "Computer Use" + scientific computing demos** — showing agents that can iterate on numerical code with feedback loops.

## Your role this week

Read, don't build. The point of this awareness pass is knowing the shape of the problem so that:

- When a paper claims "LLM beats cuBLAS," you can ask the right follow-ups (what shapes, what hardware, was it numerically correct, did the eval allow shortcuts).
- When a teammate says "we should just have an agent write our kernels," you know the answer is "for some kernels, in a specific harness, with a human reviewing." Not "no" — but not "yes" either.
- When deciding whether to specialize in compiler / kernel engineering, you can factor in the realistic trajectory: this is a field where AI assistance is real and growing, but the leverage is on the human who can architect a problem and a harness.

## Recommended reading order

1. KernelBench paper — https://arxiv.org/abs/2502.10517 — 1 hour. Sets the eval framework.
2. Sakana AI's CUDA agent post and its follow-ups — 30 minutes. Note both the methodology and the criticism.
3. One Triton autotuner blog post — pick any from PyTorch Conference 2024/2025 — 30 minutes.
4. Skim a CUTLASS production kernel (FA3 or DeepGEMM, from Topic 07) — 30 minutes. This is the level LLMs can't reach yet, and seeing it concretely calibrates expectations.

That's the 2.5-hour pass. Sufficient for awareness.

## What's actually changing in 2026

- **Frontier models can write correct Triton matmuls reliably** — not always fast, but compilable and numerically correct, on first or second try. This was unreliable in 2023.
- **Eval contamination is taken seriously now** — KernelBench and follow-ups have explicit splits to prevent training-set leakage. Numbers from before this should be treated with caution.
- **In-the-loop agents are the dominant pattern** — pure one-shot generation is rarely how production kernel work uses LLMs. The harness (compile, benchmark, error-message-back) is where the leverage is.
- **Small specialized models for specific kernel domains** is a research direction; a 7B model fine-tuned on Triton kernels can outperform a frontier general-purpose model for that narrow task. Whether this becomes the production pattern is open.
- **Compiler-pass synthesis is an active research area** but no production stack ships LLM-generated MLIR passes as of early 2026. This is the next frontier; if it lands, it changes how compiler teams work.
