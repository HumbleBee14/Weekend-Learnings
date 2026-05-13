# Notes — 05

Capture as you walk through the three files:

- Did you have B200 access? If yes: numbers for single-SM vs 2-SM at 4096³ BF16.
- The six SM100-specific lines you can point to in walkthrough 01 (TMEM alloc, accumulator view, single-thread issue, umma_arrive, tcgen05.ld, dealloc).
- One sentence on why `tcgen05.mma` is single-thread issue but `tcgen05.ld` is warpgroup-wide. (Hint: TMEM has 128 lanes; one warp sees 32.)
- One sentence on why 2-SM cooperative MMA is a loss on decode-shape M=1 GEMMs.
- After reading the three walkthroughs, open [FA4's source](https://github.com/Dao-AILab/flash-attention) and point to its TMEM accumulator and its `tcgen05.mma.cta_group::1`. Note the warp role differences from a GEMM.
