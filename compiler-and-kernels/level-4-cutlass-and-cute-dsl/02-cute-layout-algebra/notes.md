# Notes — 02

Track here as you work the worked examples:

- Examples where your hand derivation disagreed with `pytest` output (this is the most useful data).
- Patterns you noticed (e.g. "every time stride < shape, the composition crosses a mode boundary").
- A swizzle you visualized: pick `Swizzle<3,4,3>` and a 128-byte row stride, write out which bank each of 32 threads lands on, with and without the XOR.
- Two CUTLASS example files where you can now spot the layout-composition pattern (good targets: `hopper/dense_gemm_persistent.py`, `blackwell/01_mma_sm100.cu`).
