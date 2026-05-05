# Level 8 — AI-Assisted Kernel Synthesis

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: AutoKernel overnight run on your model; kernel quality review report

## Week goal

This is the frontier — LLMs writing GPU kernels. By Friday you should be able to:

- Understand what KernelBench measures and what current models (o3, Claude) can and can't do
- Run AutoKernel against your own transformer; review and evaluate the generated kernels
- Understand the 5-stage validation harness that separates AI-generated kernels from random GPU noise
- Form a clear mental model of where AI synthesis wins, where it fails, and why

## Where this fits

- **Comes after:** Levels 1–7. You need to be able to *read and evaluate* AI-generated kernels. If you can't judge a kernel's quality yourself, you can't safely deploy one. This week only works because you have the preceding skills.
- **Comes before:** Level 9 (Rust). Intentionally placed here as a "lighter" week after the heavy MLIR/StableHLO work.

## 2026 reality check

**Honest assessment of the field.**

- **Production-useful now:** AutoKernel-style agentic loops for elementwise/reduction kernels (RMSNorm, softmax, SiLU, RoPE variants). These are well-shaped problems where the LLM can generate candidates quickly and the correctness harness makes deployment safe with code review.
- **Still research:** General GEMM synthesis, attention kernel synthesis, MoE kernel synthesis. LLMs lack the hardware precision model to beat hand-tuned CUTLASS or Triton for GEMM-heavy work.
- **The trajectory:** The FlashInfer-Bench "virtuous cycle" model — AI kernels are continuously benchmarked, hot-swapped into serving engines, and the winners are kept. This changes the economics: instead of one engineer writing one kernel carefully, you run 300 experiments overnight and pick the best.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | kernelbench-baseline | Understand the benchmark; run frontier models on it |
| 02 | autokernel-architecture | The agentic loop + 5-stage validation harness |
| 03 | running-autokernel | Overnight run on your transformer; collect results |
| 04 | evaluating-ai-kernels | How to read and judge generated Triton code |
| 05 | cutegen | CuTe-DSL kernel generation; where LLMs do better |
| 06 | flashinfer-bench-model | Hot-swap architecture; virtuous cycle |
| 07 | when-to-use | Decision framework: write vs generate vs compile |

### 01 — `kernelbench-baseline`

**What KernelBench measures.** 250 PyTorch workloads, ranging from simple elementwise ops to full transformer attention. For each, an LLM generates a CUDA/Triton kernel. Scoring: correctness + speedup over the PyTorch baseline.

**Current results (2026).** Frontier reasoning models (o3, Claude 3.5 Sonnet) achieve speedup over PyTorch on fewer than 20% of problems out of the box. On simple elementwise and reduction problems they do well; on GEMM and attention problems they fail to beat cuBLAS/FlashAttention. This is the honest baseline.

**Why the number seems low.** "20% of KernelBench" includes hard problems like fused multi-head attention. A better framing: on the subset of problems where hand-written Triton is the right approach (reduction-heavy elementwise ops), LLMs do much better — often matching or exceeding human-written Triton within 5–10 iterations.

**Build steps.** Clone [KernelBench](https://github.com/ScalingIntelligence/KernelBench). Run level 1 (simple ops) with a Claude API call. Look at: which problems does it solve? Which does it fail on? What patterns distinguish the successes from failures?

### 02 — `autokernel-architecture`

**What AutoKernel does differently from naive "generate a kernel."** The agentic loop:

1. **Profile.** Run `torch.profiler` on your model. Rank bottlenecks by Amdahl's law — not just raw time, but `time × (1 - 1/speedup_ceiling)`. This ensures the agent focuses on ops that can actually be improved.

2. **Generate.** An LLM (Claude/GPT-4o) generates a Triton kernel for the bottleneck op. The prompt includes: the op's PyTorch implementation, the input shapes, the dtype, the GPU model, and examples of good Triton patterns.

3. **Validate (5 stages):**
   - Smoke test: does it compile and run on a small input?
   - Shape sweep: correct on 8 different input sizes?
   - Adversarial numerics: correct on inputs with infinities, NaNs, very large/small values?
   - Determinism: same output on two runs with same seed?
   - Non-power-of-2: correct when sizes aren't powers of 2?

4. **Benchmark.** If all 5 validation stages pass: measure throughput. If faster than the existing kernel → keep. Otherwise → discard.

5. **Iterate.** Run 40 experiments per hour, 300+ overnight. Each failed experiment feeds back into the prompt ("this kernel failed with error: ...").

**The 5-stage harness is the key innovation.** Without it, you'd deploy a kernel that works on your test case but fails on edge inputs at 3 AM. The harness catches the failures before they reach production.

### 03 — `running-autokernel`

**Setup.**
```bash
pip install autokernel
# Requires an LLM API key (Anthropic or OpenAI)
export ANTHROPIC_API_KEY=...
```

**Run on your model.**
```python
from autokernel import AutoKernel

# Point at a model you know
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

ak = AutoKernel(model=model, target_device="cuda", llm="claude-3-5-sonnet")
results = ak.run(sample_inputs, max_experiments=100)  # ~2.5 hrs
results.save("autokernel_results/")
```

**What to collect.**
- Which ops did AutoKernel target? (Amdahl ranking)
- How many experiments ran per op?
- Acceptance rate (pass 5-stage validation): typically 30–60%
- Speedup distribution of accepted kernels
- The actual Triton code for the top kernel per op

**Review the generated kernels.** For each accepted kernel: read the Triton code. Can you see *why* it's faster? Is it a fusion you didn't write in Level 5? Different tile size? Different reduction strategy? Often the best LLM-generated kernels find a non-obvious tile configuration in the autotune space.

### 04 — `evaluating-ai-kernels`

**How to judge generated Triton code.** After Level 1 you can read Triton. Use that skill here. For each generated kernel, check:

1. **Correctness coverage.** Does the kernel handle non-power-of-2 sizes? Does it mask correctly? Are boundary conditions handled?
2. **Memory access pattern.** Does it coalesce reads and writes? (Contiguous threads should access contiguous memory.)
3. **Tile size choice.** Is the tile size sensible for the GPU's SMEM budget? (SMEM per SM / bytes per tile should be ≥ 1 for the kernel to fit.)
4. **Register pressure.** Will this spill? (More than ~128 32-bit registers per thread on modern GPUs is a warning sign.)
5. **Autotune configs.** Are the autotuned configs sensible? (Did the LLM generate them, or did it hardcode a single config?)
6. **Numerical stability.** For reduction kernels (softmax, RMSNorm) — does it use the online algorithm or a naive two-pass?

**A kernel that passes all 5 validation stages but fails one of the above checks is a candidate for refinement, not deployment.** Your job is to decide: is this safe and worth using?

### 05 — `cutegen`

**CuTeGen (arxiv 2604.01489).** An LLM-based framework specifically for generating CuTe-DSL kernels. Different from AutoKernel (which targets Triton). CuTeGen provides a structured prompt template that describes the CuTe layout algebra to the LLM, significantly improving generation quality for GEMM-shaped ops.

**Results.** On activation kernels: average 1.70× speedup over PyTorch. On 2 GEMM benchmarks: beats reference implementations. The structured prompt approach (explaining TMA descriptors, WGMMA semantics, layout composition) is what makes GEMM generation viable — without that context, LLMs generate syntactically valid but semantically broken CuTe-DSL.

**What to try.** Take one of the GEMM shapes from Level 4 that your CuTe-DSL kernel didn't nail. Feed it to a CuTeGen-style prompt (available in the paper's appendix). See if the LLM finds a better tile configuration or epilogue structure than you did.

**Resources.**
- [CuTeGen paper — arxiv 2604.01489](https://arxiv.org/html/2604.01489)
- [Awesome LLM-driven kernel generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation)

### 06 — `flashinfer-bench-model`

**FlashInfer-Bench (Oct 2025).** A continuous benchmarking + hot-swap system for attention kernels. The idea: AI-generated kernels are benchmarked against the production FlashInfer kernel in a staging environment. If a generated kernel beats the production one on a specific (dtype, head_dim, seq_len) configuration, it's automatically submitted as a PR.

**The virtuous cycle:**
1. Developers submit KernelBench-style tasks to the FlashInfer problem set
2. LLM agents generate candidate kernels overnight
3. FlashInfer-Bench validates and benchmarks them
4. Winners update the production kernel
5. New benchmark baseline → harder for future LLM-generated kernels to beat → better kernels get generated

**The implication for the field.** The question "will LLMs replace kernel engineers?" misses the point. A more accurate framing: "AI-assisted synthesis + expert review" becomes a tool that expert kernel engineers use to explore the optimization space faster. The expert's role shifts from "write every kernel manually" to "judge generated kernels, identify patterns, improve the prompt."

### 07 — `when-to-use`

**The decision framework.** Given a bottleneck op, choose your approach:

```
Is the op GEMM-shaped (matmul dominant)?
  → Yes: Use CUTLASS/CuTe-DSL directly. LLMs add noise here.
  → No: Is it a standard reduction (RMSNorm, softmax, cross-entropy)?
      → Yes: Try AutoKernel first; compare to Liger-Kernel.
             If AutoKernel beats Liger-Kernel: use it with code review.
             If Liger-Kernel is already near roofline: just use Liger-Kernel.
      → No: Is it a custom attention variant?
          → Yes: Use FlexAttention. LLMs can't beat FlashInfer here.
          → No: Novel op?
              → Try AutoKernel (reasonable at genuinely novel ops)
              → Write Triton manually if AutoKernel fails validation
```

**What to put in the report.** For each op AutoKernel targeted on your model:
- Did it generate a faster kernel?
- Was the kernel safe to deploy (code review passed)?
- What did the LLM discover that you didn't think to try?
- What was wrong with the kernels that failed validation?

## Project this week

```
compiler-and-kernels/
└── ai_kernels/
    ├── kernelbench_level1.py     # KernelBench level 1 run with Claude
    ├── autokernel_run.py         # Overnight AutoKernel run on your model
    ├── kernel_review/
    │   ├── best_rmsnorm.py       # Top kernel from AutoKernel, annotated
    │   └── best_swiglu.py        # Top SwiGLU kernel, annotated
    └── reports/
        └── level8-ai-kernels.md # Results + decision framework + honest assessment
```

**Report structure:**
1. AutoKernel results summary: ops targeted, acceptance rate, speedup distribution
2. Code review of top 3 accepted kernels: what did the LLM find?
3. Failure analysis: what patterns failed validation, and why?
4. Decision framework: when you'd use AutoKernel in production (updated with your own experience)
5. One paragraph: honest assessment of where AI kernel synthesis is in 2026

## Definition of done

- [ ] You ran KernelBench level 1 and have numbers on what Claude can and can't solve.
- [ ] You ran AutoKernel on your model and collected accepted kernels.
- [ ] You reviewed at least 3 accepted kernels using the evaluation checklist from Topic 04.
- [ ] `reports/level8-ai-kernels.md` includes your results and your honest assessment.

## Resources

- **KernelBench paper** — [arxiv.org/abs/2502.10517](https://arxiv.org/abs/2502.10517).
- **KernelBench GitHub** — [github.com/ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench).
- **AutoKernel paper** — [arxiv.org/abs/2603.21331](https://arxiv.org/abs/2603.21331).
- **AutoKernel GitHub** — [github.com/RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel).
- **CuTeGen paper** — [arxiv.org/abs/2604.01489](https://arxiv.org/html/2604.01489).
- **FlashInfer-Bench blog** — [flashinfer.ai/2025/10/21/flashinfer-bench.html](https://flashinfer.ai/2025/10/21/flashinfer-bench.html).
- **Awesome LLM-driven kernel generation** — [github.com/flagos-ai/awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation).

## What you'll be able to do after this week

> Run AI-assisted kernel synthesis on a real model, evaluate the generated kernels for correctness and performance, and decide when AI synthesis adds value vs when manual Triton or CuTe-DSL is better. Understand where the field is in 2026 honestly — not hype, not dismissal.
