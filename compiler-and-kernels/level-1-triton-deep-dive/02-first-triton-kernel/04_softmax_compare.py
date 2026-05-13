"""
04 — Benchmarking discipline.

We measure four softmax implementations:
  1. eager torch.softmax
  2. our Triton kernel from 03_softmax_row.py
  3. torch.compile(torch.softmax)  — Inductor will generate its own Triton
  4. our Triton kernel under torch.compile's dispatch via a custom op (skipped here,
     covered in Level 2)

This file teaches you four habits that prevent measurement mistakes:

  a) Warmup. The first run includes JIT compile time and CUDA context setup.
     We use triton.testing.do_bench which warms up automatically.

  b) Multiple shapes. A single shape often hides a story. We sweep n_cols and
     keep n_rows fixed.

  c) Bandwidth, not just time. Same shape across implementations means same
     bytes moved — convert ms to GB/s so the numbers tell you "how close to peak".

  d) Sanity-check the output. Always compare to a known-correct reference
     before believing any timing.
"""

import torch
import triton

# Reuse the kernel from file 03.
import importlib.util
spec = importlib.util.spec_from_file_location("softmax03", "03_softmax_row.py")
softmax03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(softmax03)


def main():
    torch.manual_seed(0)
    device = "cuda"

    # torch.compile a softmax wrapper. Inductor will emit Triton for this.
    @torch.compile(mode="max-autotune", fullgraph=True)
    def compiled_softmax(x):
        return torch.softmax(x, dim=1)

    # Trigger compilation outside the timed region.
    _warmup = compiled_softmax(torch.randn(32, 512, device=device))

    results = []
    n_rows = 2048
    for n_cols in [256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(n_rows, n_cols, device=device, dtype=torch.float32)

        # Correctness check first
        out_triton = softmax03.softmax(x)
        out_torch = torch.softmax(x, dim=1)
        diff = (out_triton - out_torch).abs().max().item()
        assert diff < 1e-5, f"correctness failed at n_cols={n_cols}: {diff}"

        ms_eager = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
        ms_triton = triton.testing.do_bench(lambda: softmax03.softmax(x))
        ms_compile = triton.testing.do_bench(lambda: compiled_softmax(x))

        bytes_moved = 2 * n_rows * n_cols * 4
        gbps = lambda ms: bytes_moved / (ms * 1e-3) / 1e9

        results.append((n_cols, ms_eager, ms_triton, ms_compile, gbps(ms_eager), gbps(ms_triton), gbps(ms_compile)))

    print(f"{'n_cols':>7}  {'eager ms':>9}  {'triton ms':>10}  {'tcomp ms':>9}  "
          f"{'eager GB/s':>11}  {'triton GB/s':>12}  {'tcomp GB/s':>11}")
    print("-" * 90)
    for n_cols, me, mt, mc, ge, gt, gc in results:
        print(f"{n_cols:>7}  {me:9.3f}  {mt:10.3f}  {mc:9.3f}  "
              f"{ge:11.1f}  {gt:12.1f}  {gc:11.1f}")

    print()
    print("What you should see:")
    print("  - At small n_cols (256-1024), eager wins on launch overhead — torch ships hand-tuned")
    print("    softmax that fuses better at small sizes than naive Triton.")
    print("  - At medium-large n_cols (2048+), all three converge toward HBM peak.")
    print("    torch.compile's Inductor often matches or slightly beats hand-written Triton at this scale,")
    print("    because it autotunes (which our hand version doesn't).")
    print("  - Across all shapes, max GB/s is bounded by your GPU's HBM bandwidth.")
    print()
    print("Save this table to notes.md. We'll come back to it in sub-module 03 when")
    print("the same lesson applies to RMSNorm, except this time we'll DO the autotune ourselves.")


if __name__ == "__main__":
    main()
