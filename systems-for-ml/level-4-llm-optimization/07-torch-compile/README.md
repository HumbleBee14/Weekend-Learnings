# 07 — torch.compile for Inference

## Files

- `CONCEPTS.md` — what changed since 2024, piecewise CUDA graphs (the canonical 2026 inference recipe), how vLLM V1 uses it, gotchas
- `compare_eager_vs_compiled.py` — measure cold vs warm time for eager and two compile modes

## Quickstart

```bash
pip install torch transformers
python compare_eager_vs_compiled.py
```

## Expected output

```
config                          cold        warm       tok/s
----------------------------------------------------------------
eager (no compile)              2.1s        1980ms     50.5
torch.compile (default)         28.4s       1450ms     69.0
torch.compile (reduce-overhead) 31.7s       1180ms     84.8
```

Cold time for compiled is dominated by JIT + Inductor lowering. Warm steady-state shows the win: 1.4-1.7× over eager for decode-heavy workloads at low batch.

`reduce-overhead` mode uses CUDA graphs internally; bigger win at small batch where launch overhead dominates.

## What this gives you over `compiler-and-kernels` Level 2

That track has the *internals* — bytecode tracing, FX graph manipulation, depyf debugging, custom backends. This topic is the *practical inference recipe* — how to use torch.compile in your serving stack and what to expect.

If torch.compile breaks for you in production, go read `compiler-and-kernels` Level 2.

## Try

- **Run twice.** The second run is much faster because of the compile cache.
- **Apply to your `mini-serve` from Level 1.** Wrap `model.generate` in `torch.compile`. Measure delta.
- **Use `enforce_eager=True` in vLLM** to disable compile, then re-enable. Compare end-to-end throughput.
- **Set `--compilation-config` capture sizes** in vLLM. Try only `[1, 8]` vs `[1, 2, 4, 8, 16, 32]`. Measure cold start vs steady-state trade-off.

## Where this goes

Topic 08 is the *manual* version of what torch.compile does automatically: kernel fusion. You'll see that hand-written fused kernels (Liger-Kernel, FlashInfer) often beat torch.compile on specific patterns — and that's why production stacks combine both.
