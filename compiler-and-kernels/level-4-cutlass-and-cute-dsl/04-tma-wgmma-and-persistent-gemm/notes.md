# Notes — 04

A note on the code in this folder: the five `stageN_*.py` files are written
against the CuTe-DSL Python API shape that NVIDIA has converged on through
CUTLASS 4.x. Specific helper names (e.g. `cute.make_smem_tensor`,
`cute.create_tma_atom`, mbarrier APIs) have shifted in beta; the canonical
"runs today" reference is always the in-tree example file in the CUTLASS
repo. If a function name in this folder does not exist in your installed
DSL version, open the matching upstream example and translate — the kernel
*structure* is what matters and is stable.

For each stage record:

- Achieved TFLOPS and percent-of-cuBLAS on your hardware.
- Tensor-core utilization from `ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active`.
- HBM bandwidth utilization (`dram__throughput.avg.pct_of_peak_sustained_elapsed`).
- One sentence: which hardware mechanism this stage exploited that the previous didn't.

Stage 5 numbers go on the capstone benchmark table.
